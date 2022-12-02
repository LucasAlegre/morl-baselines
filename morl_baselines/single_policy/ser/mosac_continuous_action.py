import random
import time
from copy import deepcopy
from typing import Union, Tuple, Optional

import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mo_gym import MOSyncVectorEnv, MORecordEpisodeStatistics
from torch.utils.tensorboard import SummaryWriter

from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.morl_algorithm import MOPolicy
from morl_baselines.common.networks import mlp
from morl_baselines.common.utils import log_episode_info, polyak_update


# The implementation of this file is largely based on CleanRL's SAC implementation
# https://github.com/vwxyzjn/cleanrl/blob/28fd178ca182bd83c75ed0d49d52e235ca6cdc88/cleanrl/sac_continuous_action.py


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = MORecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class MOSoftQNetwork(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_shape,
        reward_dim,
        net_arch=[256, 256],
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.reward_dim = reward_dim
        self.net_arch = net_arch

        # S, A -> ... -> |R| (multi-objective)
        self.critic = mlp(
            input_dim=np.array(self.obs_shape).prod() + np.prod(self.action_shape),
            output_dim=self.reward_dim,
            net_arch=self.net_arch,
            activation_fn=nn.ReLU,
        )

    def forward(self, x, a):
        x = th.cat([x, a], dim=-1)
        x = self.critic(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class MOSACActor(nn.Module):
    """
    Actor network: S -> A. Does not need any multi-objective concept.
    """

    def __init__(
        self,
        obs_shape: Tuple,
        action_shape: Tuple,
        reward_dim: int,
        action_lower_bound,
        action_upper_bound,
        net_arch=[256, 256],
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.reward_dim = reward_dim
        self.net_arch = net_arch

        # S -> ... -> |A| (mean)
        #          -> |A| (std)
        self.fc1 = nn.Linear(np.array(self.obs_shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(self.action_shape))
        self.fc_logstd = nn.Linear(256, np.prod(self.action_shape))
        # action rescaling
        self.register_buffer("action_scale", th.tensor((action_upper_bound - action_lower_bound) / 2.0, dtype=th.float32))
        self.register_buffer("action_bias", th.tensor((action_upper_bound + action_lower_bound) / 2.0, dtype=th.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = th.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = th.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = th.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= th.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = th.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class MOSAC(MOPolicy):
    def __init__(
        self,
        envs: MOSyncVectorEnv,
        weights: np.ndarray,
        scalarization=th.dot,
        buffer_size: int = int(1e6),
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        learning_starts: int = int(5e3),
        net_arch=[256, 256],
        policy_lr: float = 3e-4,
        q_lr: float = 1e-3,
        policy_freq: int = 2,
        target_net_freq: int = 1,
        alpha: float = 0.2,
        autotune: bool = True,
        id: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        torch_deterministic: bool = True,
        parent_writer: Optional[SummaryWriter] = None,
        log: bool = True,
        seed: int = 42,
    ):
        """
        :param envs: Vectorized Envs
        :param weights: weights for the scalarization
        :param scalarization: scalarization
        :param buffer_size: buffer size
        :param gamma:
        :param tau: target smoothing coefficient (polyak update)
        :param batch_size:
        :param learning_starts: how many steps to collect before triggering the learning
        :param net_arch: number of nodes in the hidden layers
        :param policy_lr: learning rate of the policy
        :param q_lr: learning rate of the q networks
        :param policy_freq: the frequency of training policy (delayed)
        :param target_net_freq: the frequency of updates for the target networks
        :param alpha: Entropy regularization coefficient
        :param autotune: automatic tuning of alpha
        :param id: id of the SAC policy, for multi-policy algos
        :param device: torch device
        :param torch_deterministic:
        :param log: logging activated or not
        :param seed:
        """

        super().__init__(id, device)
        # Seeding
        self.seed = seed
        self.log = log
        self.torch_deterministic = torch_deterministic
        random.seed(self.seed)
        np.random.seed(self.seed)
        th.manual_seed(self.seed)
        th.backends.cudnn.torch_deterministic = self.torch_deterministic

        # env setup
        self.envs = envs
        self.num_envs = envs.num_envs
        assert isinstance(self.envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
        self.obs_shape = self.envs.single_observation_space.shape
        self.action_shape = self.envs.single_action_space.shape
        self.reward_dim = self.envs.reward_space.shape[0]

        # Scalarization
        self.weights = weights
        self.weights_tensor = th.from_numpy(self.weights).to(self.device).unsqueeze(1).repeat(1, self.num_envs).to(self.device)
        self.scalarization = scalarization

        # SAC Parameters
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.net_arch = net_arch
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.policy_freq = policy_freq
        self.target_net_freq = target_net_freq

        # Networks
        self.actor = MOSACActor(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            reward_dim=self.reward_dim,
            action_lower_bound=self.envs.action_space.low,
            action_upper_bound=self.envs.action_space.high,
            net_arch=self.net_arch,
        ).to(self.device)

        self.qf1 = MOSoftQNetwork(
            obs_shape=self.obs_shape, action_shape=self.action_shape, reward_dim=self.reward_dim, net_arch=self.net_arch
        ).to(self.device)
        self.qf2 = MOSoftQNetwork(
            obs_shape=self.obs_shape, action_shape=self.action_shape, reward_dim=self.reward_dim, net_arch=self.net_arch
        ).to(self.device)
        self.qf1_target = MOSoftQNetwork(
            obs_shape=self.obs_shape, action_shape=self.action_shape, reward_dim=self.reward_dim, net_arch=self.net_arch
        ).to(self.device)
        self.qf2_target = MOSoftQNetwork(
            obs_shape=self.obs_shape, action_shape=self.action_shape, reward_dim=self.reward_dim, net_arch=self.net_arch
        ).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.policy_lr)

        # Automatic entropy tuning
        self.autotune = autotune
        if self.autotune:
            self.target_entropy = -th.prod(th.Tensor(envs.single_action_space.shape).to(self.device)).item()
            self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr)
        else:
            self.alpha = alpha

        # Buffer
        self.envs.single_observation_space.dtype = np.float32
        self.buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            action_dim=self.action_shape[0],
            rew_dim=self.reward_dim,
            num_envs=self.num_envs,
            max_size=self.buffer_size,
        )

        # Logging
        self.writer = parent_writer

    def __deepcopy__(self, memo):

        copied = type(self)(
            envs=self.envs,
            weights=self.weights,
            scalarization=self.scalarization,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            tau=self.tau,
            batch_size=self.batch_size,
            learning_starts=self.learning_starts,
            net_arch=self.net_arch,
            policy_lr=self.policy_lr,
            q_lr=self.q_lr,
            policy_freq=self.policy_freq,
            target_net_freq=self.target_net_freq,
            alpha=self.alpha,
            autotune=self.autotune,
            id=self.id,
            device=self.device,
            torch_deterministic=self.torch_deterministic,
            parent_writer=self.writer,
            log=self.log,
            seed=self.seed,
        )

        # Copying networks
        copied.actor = deepcopy(self.actor)
        copied.qf1 = deepcopy(self.qf1)
        copied.qf2 = deepcopy(self.qf2)
        copied.qf1_target = deepcopy(self.qf1_target)
        copied.qf2_target = deepcopy(self.qf2_target)

        copied.global_step = self.global_step
        copied.actor_optimizer = optim.Adam(copied.actor.parameters(), lr=self.policy_lr, eps=1e-5)
        copied.q_optimizer = optim.Adam(list(copied.qf1.parameters()) + list(copied.qf2.parameters()), lr=self.q_lr)
        copied.a_optimizer = optim.Adam(list(copied.qf1.parameters()) + list(copied.qf2.parameters()), lr=self.q_lr)

        if self.autotune:
            optim.Adam([copied.log_alpha], lr=self.q_lr)
        copied.buffer = deepcopy(self.buffer)
        return copied

    def get_buffer(self):
        return self.buffer

    def set_buffer(self, buffer):
        self.buffer = buffer

    def get_policy_net(self) -> th.nn.Module:
        return self.actor

    def eval(self, obs: np.ndarray, w: Optional[np.ndarray]) -> Union[int, np.ndarray]:
        """
        Returns the best action to perform for the given obs
        :return: action as a numpy array (continuous actions)
        """
        obs = th.as_tensor(obs).float().to(self.device)
        obs = obs.unsqueeze(0).repeat(self.num_envs, 1)  # duplicate observation to fit the NN input
        with th.no_grad():
            action, _, _ = self.actor.get_action(obs)

        return action[0].detach().cpu().numpy()

    def update(self):
        (mb_obs, mb_act, mb_rewards, mb_next_obs, mb_dones) = self.buffer.sample(
            self.batch_size, to_tensor=True, device=self.device
        )

        # TODO maybe we can avoid scalarizing next_q_value?
        with th.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(mb_next_obs)
            # Q values are scalarized before being compared (min of ensemble networks)
            qf1_next_target = self.scalarization(self.qf1_target(mb_next_obs, next_state_actions), self.weights_tensor)
            qf2_next_target = self.scalarization(self.qf2_target(mb_next_obs, next_state_actions), self.weights_tensor)
            min_qf_next_target = th.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            scalarized_rewards = self.scalarization(mb_rewards, self.weights_tensor)
            next_q_value = scalarized_rewards.flatten() + (1 - mb_dones.flatten()) * self.gamma * min_qf_next_target.view(-1)

        qf1_a_values = self.scalarization(self.qf1(mb_obs, mb_act), self.weights_tensor).flatten()
        qf2_a_values = self.scalarization(self.qf2(mb_obs, mb_act), self.weights_tensor).flatten()
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if self.global_step % self.policy_freq == 0:  # TD 3 Delayed update support
            for _ in range(self.policy_freq):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = self.actor.get_action(mb_obs)
                # Q values are scalarized before being compared (min of ensemble networks)
                qf1_pi = self.scalarization(self.qf1(mb_obs, pi), self.weights_tensor)
                qf2_pi = self.scalarization(self.qf2(mb_obs, pi), self.weights_tensor)
                min_qf_pi = th.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune:
                    with th.no_grad():
                        _, log_pi, _ = self.actor.get_action(mb_obs)
                    alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        # update the target networks
        if self.global_step % self.target_net_freq == 0:
            polyak_update(params=self.qf1.parameters(), target_params=self.qf1_target.parameters(), tau=self.tau)
            polyak_update(params=self.qf2.parameters(), target_params=self.qf2_target.parameters(), tau=self.tau)

        if self.global_step % 100 == 0:
            log_str = f"_{self.id}" if self.id is not None else ""
            self.writer.add_scalar(f"losses{log_str}/qf1_values", qf1_a_values.mean().item(), self.global_step)
            self.writer.add_scalar(f"losses{log_str}/qf2_values", qf2_a_values.mean().item(), self.global_step)
            self.writer.add_scalar(f"losses{log_str}/qf1_loss", qf1_loss.item(), self.global_step)
            self.writer.add_scalar(f"losses{log_str}/qf2_loss", qf2_loss.item(), self.global_step)
            self.writer.add_scalar(f"losses{log_str}/qf_loss", qf_loss.item() / 2.0, self.global_step)
            self.writer.add_scalar(f"losses{log_str}/actor_loss", actor_loss.item(), self.global_step)
            self.writer.add_scalar(f"losses{log_str}/alpha", self.alpha, self.global_step)
            if self.autotune:
                self.writer.add_scalar(f"losses{log_str}/alpha_loss", alpha_loss.item(), self.global_step)

    def train(self, total_timesteps: int, eval_env: Optional[gym.Env] = None):
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = self.envs.reset(seed=self.seed)
        for step in range(total_timesteps):
            # ALGO LOGIC: put action logic here
            if self.global_step < self.learning_starts:
                actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            else:
                actions, _, _ = self.actor.get_action(th.Tensor(obs).to(self.device))
                actions = actions.detach().cpu().numpy()

            # execute the game and log data
            next_obs, rewards, terminateds, truncateds, infos = self.envs.step(actions)

            # Episode info logging
            if "final_info" in infos.keys():
                _info = infos["final_info"][0]["episode"]
                log_episode_info(
                    _info,
                    np.dot,
                    self.weights,
                    self.global_step,
                    self.id,
                    self.writer,
                )

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs
            if "final_observation" in infos:
                real_next_obs = next_obs.copy()
                for idx, d in enumerate(infos["_final_observation"]):
                    if d:
                        real_next_obs[idx] = infos["final_observation"][idx]
            self.buffer.add(
                obs=obs, next_obs=real_next_obs, action=actions, reward=rewards, done=np.expand_dims(terminateds, axis=1)
            )

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if self.global_step > self.learning_starts:
                self.update()
                if self.global_step % 100 == 0:
                    print("SPS:", int(self.global_step / (time.time() - start_time)))
                    self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - start_time)), self.global_step)

            self.global_step += self.num_envs

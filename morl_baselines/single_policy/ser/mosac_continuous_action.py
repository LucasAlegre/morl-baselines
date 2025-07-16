"""Multi-objective Soft Actor-Critic (SAC) algorithm for continuous action spaces.

It implements a multi-objective critic with weighted sum scalarization.
The implementation of this file is largely based on CleanRL's SAC implementation
https://github.com/vwxyzjn/cleanrl/blob/28fd178ca182bd83c75ed0d49d52e235ca6cdc88/cleanrl/sac_continuous_action.py
"""

import time
from copy import deepcopy
from typing import Optional, Tuple, Union
from typing_extensions import override

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.evaluation import log_episode_info
from morl_baselines.common.morl_algorithm import MOPolicy
from morl_baselines.common.networks import layer_init, mlp, polyak_update


# ALGO LOGIC: initialize agent here:
class MOSoftQNetwork(nn.Module):
    """Soft Q-network: S, A -> ... -> |R| (multi-objective)."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        reward_dim,
        net_arch=[256, 256],
    ):
        """Initialize the soft Q-network."""
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
        self.apply(layer_init)

    def forward(self, x, a):
        """Forward pass of the soft Q-network."""
        x = th.cat([x, a], dim=-1)
        x = self.critic(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class MOSACActor(nn.Module):
    """Actor network: S -> A. Does not need any multi-objective concept."""

    def __init__(
        self,
        obs_shape: Tuple,
        action_shape: Tuple,
        reward_dim: int,
        action_lower_bound,
        action_upper_bound,
        net_arch=[256, 256],
    ):
        """Initialize SAC actor."""
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.reward_dim = reward_dim
        self.net_arch = net_arch

        # S -> ... -> |A| (mean)
        #          -> |A| (std)
        self.latent_pi = mlp(np.array(self.obs_shape).prod(), -1, self.net_arch)
        self.fc_mean = nn.Linear(net_arch[-1], np.prod(self.action_shape))
        self.fc_logstd = nn.Linear(net_arch[-1], np.prod(self.action_shape))
        self.apply(layer_init)
        # action rescaling
        self.register_buffer(
            "action_scale",
            th.tensor((action_upper_bound - action_lower_bound) / 2.0, dtype=th.float32),
        )
        self.register_buffer(
            "action_bias",
            th.tensor((action_upper_bound + action_lower_bound) / 2.0, dtype=th.float32),
        )

    def forward(self, x):
        """Forward pass of the actor network."""
        x = self.latent_pi(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = th.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        """Get action from the actor network."""
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
    """Multi-objective Soft Actor-Critic (SAC) algorithm.

    It is a multi-objective version of the SAC algorithm, with multi-objective critic and weighted sum scalarization.
    """

    def __init__(
        self,
        env: gym.Env,
        weights: np.ndarray,
        scalarization=th.matmul,
        buffer_size: int = int(1e6),
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 128,
        learning_starts: int = int(1e3),
        net_arch=[256, 256],
        policy_lr: float = 3e-4,
        q_lr: float = 1e-3,
        policy_freq: int = 2,
        target_net_freq: int = 1,
        alpha: float = 0.2,
        autotune: bool = True,
        id: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        log: bool = True,
        seed: int = 42,
        parent_rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the MOSAC algorithm.

        Args:
            env: Env
            weights: weights for the scalarization
            scalarization: scalarization function
            buffer_size: buffer size
            gamma: discount factor
            tau: target smoothing coefficient (polyak update)
            batch_size: batch size
            learning_starts: how many steps to collect before triggering the learning
            net_arch: number of nodes in the hidden layers
            policy_lr: learning rate of the policy
            q_lr: learning rate of the q networks
            policy_freq: the frequency of training policy (delayed)
            target_net_freq: the frequency of updates for the target networks
            alpha: Entropy regularization coefficient
            autotune: automatic tuning of alpha
            id: id of the SAC policy, for multi-policy algos
            device: torch device
            torch_deterministic: whether to use deterministic version of pytorch
            log: logging activated or not
            seed: seed for the random generators
            parent_rng: parent random generator, for multi-policy algos
        """
        super().__init__(id, device)
        # Seeding
        self.seed = seed
        self.parent_rng = parent_rng
        if parent_rng is not None:
            self.np_random = parent_rng
        else:
            self.np_random = np.random.default_rng(self.seed)

        # env setup
        self.env = env
        assert isinstance(self.env.action_space, gym.spaces.Box), "only continuous action space is supported"
        self.obs_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.reward_dim = self.env.unwrapped.reward_space.shape[0]

        # Scalarization
        self.weights = weights
        self.weights_tensor = th.from_numpy(self.weights).float().to(self.device)
        self.batch_size = batch_size
        self.scalarization = scalarization

        # SAC Parameters
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
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
            action_lower_bound=self.env.action_space.low,
            action_upper_bound=self.env.action_space.high,
            net_arch=self.net_arch,
        ).to(self.device)

        self.qf1 = MOSoftQNetwork(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            reward_dim=self.reward_dim,
            net_arch=self.net_arch,
        ).to(self.device)
        self.qf2 = MOSoftQNetwork(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            reward_dim=self.reward_dim,
            net_arch=self.net_arch,
        ).to(self.device)
        self.qf1_target = MOSoftQNetwork(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            reward_dim=self.reward_dim,
            net_arch=self.net_arch,
        ).to(self.device)
        self.qf2_target = MOSoftQNetwork(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            reward_dim=self.reward_dim,
            net_arch=self.net_arch,
        ).to(self.device)
        self.qf1_target.requires_grad_(False)
        self.qf2_target.requires_grad_(False)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.policy_lr)

        # Automatic entropy tuning
        self.autotune = autotune
        if self.autotune:
            self.target_entropy = -th.prod(th.Tensor(env.action_space.shape).to(self.device)).item()
            self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr)
        else:
            self.alpha = alpha
        self.alpha_tensor = th.scalar_tensor(self.alpha).to(self.device)

        # Buffer
        self.env.observation_space.dtype = np.float32
        self.buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            action_dim=self.action_shape[0],
            rew_dim=self.reward_dim,
            max_size=self.buffer_size,
        )

        # Logging
        self.log = log

    def get_config(self) -> dict:
        """Returns the configuration of the policy."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "net_arch": self.net_arch,
            "policy_lr": self.policy_lr,
            "q_lr": self.q_lr,
            "policy_freq": self.policy_freq,
            "target_net_freq": self.target_net_freq,
            "alpha": self.alpha,
            "autotune": self.autotune,
            "seed": self.seed,
        }

    def __deepcopy__(self, memo):
        """Deep copy of the policy.

        Args:
            memo (dict): memoization dict
        """
        copied = type(self)(
            env=self.env,
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
            log=self.log,
            seed=self.seed,
            parent_rng=self.parent_rng,
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
        if self.autotune:
            copied.a_optimizer = optim.Adam([copied.log_alpha], lr=self.q_lr)
        copied.alpha_tensor = th.scalar_tensor(copied.alpha).to(self.device)
        copied.buffer = deepcopy(self.buffer)
        return copied

    @override
    def get_buffer(self):
        return self.buffer

    @override
    def set_buffer(self, buffer):
        self.buffer = buffer

    @override
    def get_policy_net(self) -> th.nn.Module:
        return self.actor

    @override
    def set_weights(self, weights: np.ndarray):
        self.weights = weights
        self.weights_tensor = th.from_numpy(self.weights).float().to(self.device)

    @override
    def get_save_dict(self, save_replay_buffer: bool = False) -> dict:
        """Returns a dictionary of all components needed for saving the MOSAC instance."""
        save_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "qf1_state_dict": self.qf1.state_dict(),
            "qf2_state_dict": self.qf2.state_dict(),
            "qf1_target_state_dict": self.qf1_target.state_dict(),
            "qf2_target_state_dict": self.qf2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "weights": self.weights,
            "alpha": self.alpha,
        }

        if save_replay_buffer:
            save_dict["buffer"] = self.buffer

        if self.autotune:  # previously used autotune
            save_dict["log_alpha"] = self.log_alpha
            save_dict["a_optimizer_state_dict"] = self.a_optimizer.state_dict()

        return save_dict

    @override
    def load(
        self,
        save_dict: Optional[dict] = None,
        path: Optional[str] = None,
        load_replay_buffer: bool = True,
    ):
        """Load the model and the replay buffer if specified."""
        if save_dict is None:
            assert path is not None, "Either save_dict or path should be provided."
            save_dict = th.load(path, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(save_dict["actor_state_dict"])
        self.qf1.load_state_dict(save_dict["qf1_state_dict"])
        self.qf2.load_state_dict(save_dict["qf2_state_dict"])
        self.qf1_target.load_state_dict(save_dict["qf1_target_state_dict"])
        self.qf2_target.load_state_dict(save_dict["qf2_target_state_dict"])
        self.actor_optimizer.load_state_dict(save_dict["actor_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(save_dict["q_optimizer_state_dict"])

        if "log_alpha" in save_dict:
            self.log_alpha = save_dict["log_alpha"]
            self.a_optimizer.load_state_dict(save_dict["a_optimizer_state_dict"])

        if load_replay_buffer:
            self.buffer = save_dict["buffer"]

        self.weights = save_dict["weights"]
        self.alpha = save_dict["alpha"]

    @override
    def eval(self, obs: np.ndarray, w: Optional[np.ndarray] = None) -> Union[int, np.ndarray]:
        """Returns the best action to perform for the given obs.

        Args:
            obs: observation as a numpy array
            w: None
        Return:
            action as a numpy array (continuous actions)
        """
        obs = th.as_tensor(obs).float().to(self.device)
        obs = obs.unsqueeze(0)
        with th.no_grad():
            action, _, _ = self.actor.get_action(obs)

        return action[0].detach().cpu().numpy()

    @override
    def update(self):
        (mb_obs, mb_act, mb_rewards, mb_next_obs, mb_dones) = self.buffer.sample(
            self.batch_size, to_tensor=True, device=self.device
        )

        with th.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(mb_next_obs)
            # (!) Q values are scalarized before being compared (min of ensemble networks)
            qf1_next_target = self.scalarization(self.qf1_target(mb_next_obs, next_state_actions), self.weights_tensor)
            qf2_next_target = self.scalarization(self.qf2_target(mb_next_obs, next_state_actions), self.weights_tensor)
            min_qf_next_target = th.min(qf1_next_target, qf2_next_target) - (self.alpha_tensor * next_state_log_pi).flatten()
            scalarized_rewards = self.scalarization(mb_rewards, self.weights_tensor)
            next_q_value = scalarized_rewards.flatten() + (1 - mb_dones.flatten()) * self.gamma * min_qf_next_target

        qf1_a_values = self.scalarization(self.qf1(mb_obs, mb_act), self.weights_tensor).flatten()
        qf2_a_values = self.scalarization(self.qf2(mb_obs, mb_act), self.weights_tensor).flatten()
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        self.q_optimizer.step()

        if self.global_step % self.policy_freq == 0:  # TD 3 Delayed update support
            for _ in range(self.policy_freq):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = self.actor.get_action(mb_obs)
                # (!) Q values are scalarized before being compared (min of ensemble networks)
                qf1_pi = self.scalarization(self.qf1(mb_obs, pi), self.weights_tensor)
                qf2_pi = self.scalarization(self.qf2(mb_obs, pi), self.weights_tensor)
                min_qf_pi = th.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = ((self.alpha_tensor * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune:
                    with th.no_grad():
                        _, log_pi, _ = self.actor.get_action(mb_obs)
                    alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha_tensor = self.log_alpha.exp()
                    self.alpha = self.log_alpha.exp().item()

        # update the target networks
        if self.global_step % self.target_net_freq == 0:
            polyak_update(
                params=self.qf1.parameters(),
                target_params=self.qf1_target.parameters(),
                tau=self.tau,
            )
            polyak_update(
                params=self.qf2.parameters(),
                target_params=self.qf2_target.parameters(),
                tau=self.tau,
            )
            self.qf1_target.requires_grad_(False)
            self.qf2_target.requires_grad_(False)

        if self.global_step % 100 == 0 and self.log:
            log_str = f"_{self.id}" if self.id is not None else ""
            to_log = {
                f"losses{log_str}/alpha": self.alpha,
                f"losses{log_str}/qf1_values": qf1_a_values.mean().item(),
                f"losses{log_str}/qf2_values": qf2_a_values.mean().item(),
                f"losses{log_str}/qf1_loss": qf1_loss.item(),
                f"losses{log_str}/qf2_loss": qf2_loss.item(),
                f"losses{log_str}/qf_loss": qf_loss.item() / 2.0,
                f"losses{log_str}/actor_loss": actor_loss.item(),
                "global_step": self.global_step,
            }
            if self.autotune:
                to_log[f"losses{log_str}/alpha_loss"] = alpha_loss.item()
            wandb.log(to_log)

    def train(self, total_timesteps: int, eval_env: Optional[gym.Env] = None, start_time=None):
        """Train the agent.

        Args:
            total_timesteps (int): Total number of timesteps (env steps) to train for
            eval_env (Optional[gym.Env]): Gym environment used for evaluation.
            start_time (Optional[float]): Starting time for the training procedure. If None, it will be set to the current time.
        """
        if start_time is None:
            start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = self.env.reset()
        for step in range(total_timesteps):
            # ALGO LOGIC: put action logic here
            if self.global_step < self.learning_starts:
                actions = self.env.action_space.sample()
            else:
                th_obs = th.as_tensor(obs).float().to(self.device)
                th_obs = th_obs.unsqueeze(0)
                actions, _, _ = self.actor.get_action(th_obs)
                actions = actions[0].detach().cpu().numpy()

            # execute the game and log data
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs
            if "final_observation" in infos:
                real_next_obs = infos["final_observation"]
            self.buffer.add(
                obs=obs,
                next_obs=real_next_obs,
                action=actions,
                reward=rewards,
                done=terminated,
            )

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            if terminated or truncated:
                obs, _ = self.env.reset()
                if self.log and "episode" in infos.keys():
                    log_episode_info(
                        infos["episode"],
                        np.dot,
                        self.weights,
                        self.global_step,
                        self.id,
                    )

            # ALGO LOGIC: training.
            if self.global_step > self.learning_starts:
                self.update()
                if self.log and self.global_step % 100 == 0:
                    print("SPS:", int(self.global_step / (time.time() - start_time)))
                    wandb.log(
                        {
                            "charts/SPS": int(self.global_step / (time.time() - start_time)),
                            "global_step": self.global_step,
                        }
                    )

            self.global_step += 1

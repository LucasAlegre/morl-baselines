import time
import typing
from copy import deepcopy
from typing import List, Optional, Union, Callable, OrderedDict

import gym
import numpy as np
import torch as th
import torch.nn
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from morl_baselines.common.accrued_reward_buffer import AccruedRewardReplayBuffer
from morl_baselines.common.morl_algorithm import MOPolicy, MOAgent
from morl_baselines.common.networks import mlp
from torch.utils.tensorboard import SummaryWriter
from morl_baselines.common.utils import layer_init, log_episode_info


# EUPG is an ESR algorithm based on Policy Gradient (REINFORCE like)
# The idea is to condition the network on the accrued reward and to
# Scalarize the rewards based on the episodic return (accrued + future rewards)
# Paper: D. Roijers, D. Steckelmacher, and A. Nowe, Multi-objective Reinforcement Learning for the Expected Utility of the Return. 2018.


class PolicyNet(nn.Module):
    def __init__(self, obs_shape, action_dim, rew_dim, net_arch):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.rew_dim = rew_dim

        # Conditioned on accrued reward, so input takes reward
        input_dim = obs_shape[0] + rew_dim

        # |S|+|R| -> ... -> |A|
        self.net = mlp(input_dim, action_dim, net_arch, activation_fn=nn.Tanh)
        self.apply(layer_init)

    def forward(self, obs: th.Tensor, acc_reward: th.Tensor):
        input = th.cat((obs, acc_reward), dim=acc_reward.dim() - 1)
        pi = self.net(input)
        # Normalized sigmoid
        x_exp = th.sigmoid(pi)
        probas = x_exp / th.sum(x_exp)
        return probas.view(-1, self.action_dim)  # Batch Size x |Actions|

    def distribution(self, obs: th.Tensor, acc_reward: th.Tensor):
        probas = self.forward(obs, acc_reward)
        distribution = Categorical(probas)
        return distribution


class EUPG(MOPolicy, MOAgent):
    def __init__(
        self,
        env: gym.Env,
        scalarization: Callable[[np.ndarray, np.ndarray], float],
        weights: np.ndarray = np.ones(2),
        id: Optional[int] = None,
        buffer_size: int = int(1e5),
        net_arch: List = [50],
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "EUPG",
        log: bool = True,
        log_every: int = 100,
        parent_writer: Optional[SummaryWriter] = None,
        device: Union[th.device, str] = "auto",
    ):
        MOAgent.__init__(self, env, device)
        MOPolicy.__init__(self, None, device)

        self.env = env
        # EUPG is sometimes launched with vectorized environments (for example when used in outer loop settings)
        # This allows to unwrap the environment since EUPG does not work in such settings
        if isinstance(self.env, gym.vector.VectorEnv):
            self.env = env.envs[0]
        else:
            self.env = env
        self.id = id
        # RL
        self.scalarization = scalarization
        self.weights = weights
        self.gamma = gamma

        # Learning
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_rate = learning_rate
        self.buffer = AccruedRewardReplayBuffer(
            obs_shape=self.observation_shape,
            action_shape=self.action_shape,
            rew_dim=self.reward_dim,
            max_size=self.buffer_size,
            obs_dtype=np.int32,
            action_dtype=np.int32,
        )
        self.net = PolicyNet(
            obs_shape=self.observation_shape,
            rew_dim=self.reward_dim,
            action_dim=self.action_dim,
            net_arch=self.net_arch,
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log
        self.log_every = log_every
        if parent_writer is not None:
            self.writer = parent_writer
        if log and parent_writer is None:
            self.setup_wandb(self.project_name, self.experiment_name)

    def __deepcopy__(self, memo):
        copied_net = deepcopy(self.net)
        copied = type(self)(
            self.env,
            self.scalarization,
            self.weights,
            self.id,
            self.buffer_size,
            self.net_arch,
            self.gamma,
            self.learning_rate,
            self.project_name,
            self.experiment_name,
            self.log,
            parent_writer=self.writer,
            device=self.device,
        )

        copied.global_step = self.global_step
        copied.optimizer = optim.Adam(copied_net.parameters(), lr=self.learning_rate)
        copied.buffer = deepcopy(self.buffer)
        return copied

    def get_policy_net(self) -> torch.nn.Module:
        return self.net

    def get_buffer(self):
        return self.buffer

    def set_buffer(self, buffer):
        raise Exception("On-policy algorithms should not share buffer.")

    def set_weights(self, weights: np.ndarray):
        self.weights = weights

    @th.no_grad()
    def eval(self, obs: np.ndarray, accrued_reward: Optional[np.ndarray]) -> Union[int, np.ndarray]:
        if type(obs) is int:
            obs = th.as_tensor([obs]).to(self.device)
        else:
            obs = th.as_tensor(obs).to(self.device)
        accrued_reward = th.as_tensor(accrued_reward).float().to(self.device)
        probas = self.net(obs, accrued_reward)
        greedy_act = th.argmax(probas)
        return greedy_act.detach().item()

    @th.no_grad()
    def choose_action(self, obs: th.Tensor, accrued_reward: th.Tensor) -> int:
        action = self.net.distribution(obs, accrued_reward)
        action = action.sample().detach().item()
        return action

    def update(self):
        (
            obs,
            accrued_rewards,
            actions,
            rewards,
            next_obs,
            terminateds,
        ) = self.buffer.get_all_data(to_tensor=True, device=self.device)
        # Scalarized episodic reward, our target :-)
        episodic_return = th.sum(rewards, dim=0)
        scalarized_return = self.scalarization(self.weights, episodic_return.cpu().numpy())
        scalarized_return = th.scalar_tensor(scalarized_return).to(self.device)

        # For each sample in the batch, get the distribution over actions
        current_distribution = self.net.distribution(obs, accrued_rewards)
        # Policy gradient
        log_probs = current_distribution.log_prob(actions.flatten())
        loss = -th.mean(log_probs * scalarized_return)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.log:
            log_str = f"_{self.id}" if self.id is not None else ""
            self.writer.add_scalar(f"losses{log_str}/loss", loss, self.global_step)
            self.writer.add_scalar(
                f"metrics{log_str}/scalarized_episodic_return",
                scalarized_return,
                self.global_step,
            )

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 1000,
    ):
        start_time = time.time()
        # Init
        (
            obs,
            _,
        ) = self.env.reset()
        accrued_reward_tensor = th.zeros(self.reward_dim, dtype=th.float32).float().to(self.device)

        # Training loop
        for _ in range(1, total_timesteps + 1):
            self.global_step += 1

            with th.no_grad():
                # For training, takes action randomly according to the policy
                action = self.choose_action(th.Tensor(obs).to(self.device), accrued_reward_tensor)
            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)

            # Memory update
            self.buffer.add(obs, accrued_reward_tensor, action, vec_reward, next_obs, terminated)
            accrued_reward_tensor += th.from_numpy(vec_reward).to(self.device)

            if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                self.policy_eval_esr(eval_env, scalarization=self.scalarization, weights=self.weights, writer=self.writer)

            if terminated or truncated:
                # NN is updated at the end of each episode
                self.update()
                self.buffer.cleanup()
                obs, _ = self.env.reset()
                self.num_episodes += 1
                accrued_reward_tensor = th.zeros(self.reward_dim).float().to(self.device)

                if self.log and self.num_episodes % self.log_every == 0 and "episode" in info.keys():
                    log_episode_info(
                        info=info["episode"],
                        scalarization=self.scalarization,
                        weights=self.weights,
                        id=self.id,
                        global_timestep=self.global_step,
                        writer=self.writer,
                    )

            else:
                obs = next_obs

            if self.global_step % 1000 == 0:
                print("SPS:", int(self.global_step / (time.time() - start_time)))
                self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - start_time)), self.global_step)

    def get_config(self) -> dict:
        return {
            # "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
        }

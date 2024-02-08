"""EUPG is an ESR algorithm based on Policy Gradient (REINFORCE like)."""
from typing import List, Optional, Union
from typing_extensions import override

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.distributions import Categorical

from morl_baselines.common.accrued_reward_buffer import AccruedRewardReplayBuffer
from morl_baselines.common.evaluation import log_episode_info
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.networks import layer_init, mlp


class PolicyNet(nn.Module):
    """Policy network."""

    def __init__(self, obs_shape, action_dim, rew_dim, net_arch):
        """Initialize the policy network.

        Args:
            obs_shape: Observation shape
            action_dim: Action dimension
            rew_dim: Reward dimension
            net_arch: Number of units per layer
        """
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
        """Forward pass.

        Args:
            obs: Observation
            acc_reward: accrued reward

        Returns: Probability of each action

        """
        input = th.cat((obs, acc_reward), dim=acc_reward.dim() - 1)
        pi = self.net(input)
        # Normalized sigmoid
        x_exp = th.sigmoid(pi)
        probas = x_exp / th.sum(x_exp)
        return probas.view(-1, self.action_dim)  # Batch Size x |Actions|

    def distribution(self, obs: th.Tensor, acc_reward: th.Tensor):
        """Categorical distribution based on the action probabilities.

        Args:
            obs: observation
            acc_reward: accrued reward

        Returns: action distribution.

        """
        probas = self.forward(obs, acc_reward)
        distribution = Categorical(probas)
        return distribution


class EUPG(MOPolicy, MOAgent):
    """Expected Utility Policy Gradient Algorithm.

    The idea is to condition the network on the accrued reward and to scalarize the rewards based on the episodic return (accrued + future rewards)
    Paper: D. Roijers, D. Steckelmacher, and A. Nowe, Multi-objective Reinforcement Learning for the Expected Utility of the Return. 2018.
    """

    def __init__(
        self,
        env: gym.Env,
        scalarization,
        buffer_size: int = int(1e5),
        net_arch: List = [50],
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "EUPG",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
    ):
        """Initialize the EUPG algorithm.

        Args:
            env: Environment
            scalarization: Scalarization function to use (can be non-linear)
            buffer_size: Size of the replay buffer
            net_arch: Number of units per layer
            gamma: Discount factor
            learning_rate: Learning rate (alpha)
            project_name: Name of the project (for logging)
            experiment_name: Name of the experiment (for logging)
            wandb_entity: Entity to use for wandb
            log: Whether to log or not
            device: Device to use for NN. Can be "cpu", "cuda" or "auto".
            seed: Seed for the random number generator
        """
        MOAgent.__init__(self, env, device, seed=seed)
        MOPolicy.__init__(self, None, device)

        self.env = env
        # RL
        self.scalarization = scalarization
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
        ).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log
        if log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    @override
    def eval(self, obs: np.ndarray, accrued_reward: Optional[np.ndarray]) -> Union[int, np.ndarray]:
        if type(obs) is int:
            obs = th.as_tensor([obs]).to(self.device)
        else:
            obs = th.as_tensor(obs).to(self.device)
        accrued_reward = th.as_tensor(accrued_reward).float().to(self.device)
        return self.__choose_action(obs, accrued_reward)

    @th.no_grad()
    def __choose_action(self, obs: th.Tensor, accrued_reward: th.Tensor) -> int:
        action = self.net.distribution(obs, accrued_reward).sample().detach().item()
        return action

    @override
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
        scalarized_return = self.scalarization(episodic_return)

        # For each sample in the batch, get the distribution over actions
        current_distribution = self.net.distribution(obs, accrued_rewards)
        # Policy gradient
        log_probs = current_distribution.log_prob(actions)
        loss = -th.mean(log_probs * scalarized_return)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.log:
            wandb.log(
                {
                    "losses/loss": loss,
                    "metrics/scalarized_episodic_return": scalarized_return,
                    "global_step": self.global_step,
                },
            )

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 1000,
    ):
        """Train the agent.

        Args:
            total_timesteps: Number of timesteps to train for
            eval_env: Environment to run policy evaluation on
            eval_freq: Frequency of policy evaluation
        """
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
                action = self.__choose_action(th.Tensor([obs]).to(self.device), accrued_reward_tensor)
            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)

            # Memory update
            self.buffer.add(obs, accrued_reward_tensor.cpu().numpy(), action, vec_reward, next_obs, terminated)
            accrued_reward_tensor += th.from_numpy(vec_reward).to(self.device)

            if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                self.policy_eval_esr(eval_env, scalarization=self.scalarization, log=self.log)

            if terminated or truncated:
                # NN is updated at the end of each episode
                self.update()
                self.buffer.cleanup()
                obs, _ = self.env.reset()
                self.num_episodes += 1
                accrued_reward_tensor = th.zeros(self.reward_dim).float().to(self.device)

                if self.log and "episode" in info.keys():
                    log_episode_info(
                        info["episode"],
                        scalarization=self.scalarization,
                        weights=None,
                        global_timestep=self.global_step,
                    )

            else:
                obs = next_obs

    @override
    def get_config(self) -> dict:
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "seed": self.seed,
        }

"""Multi-objective Soft Actor-Critic (SAC) algorithm for discrete action spaces.

Implemented for the paper: https://arxiv.org/abs/2503.00799.

It implements a multi-objective critic with weighted sum scalarization.
The implementation of this file is largely based on CleanRL's SAC implementation
https://github.com/vwxyzjn/cleanrl/blob/28fd178ca182bd83c75ed0d49d52e235ca6cdc88/cleanrl/sac_atari.py
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
from torch.distributions.categorical import Categorical

from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.evaluation import log_episode_info
from morl_baselines.common.morl_algorithm import MOPolicy
from morl_baselines.common.networks import (
    NatureCNN,
    layer_init,
    mlp,
    polyak_update,
)


# ALGO LOGIC: initialize agent here:
class MODiscreteSoftQNetwork(nn.Module):
    """Soft Q-network: S -> ... -> |A| * |R| (multi-objective)."""

    def __init__(self, obs_shape, action_dim, reward_dim, net_arch):
        """Initialize the Q network.

        Args:
            obs_shape: shape of the observation
            action_dim: number of actions
            reward_dim: number of objectives
            net_arch: network architecture (number of units per layer)
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        if len(obs_shape) == 1:
            self.feature_extractor = mlp(
                input_dim=obs_shape[0],
                output_dim=-1,
                net_arch=net_arch[:1],
            )
        elif len(obs_shape) > 1:  # Image observation
            self.feature_extractor = NatureCNN(self.obs_shape, features_dim=net_arch[0])
        #  ... -> |A| * |R|
        self.net = mlp(
            input_dim=net_arch[0],
            output_dim=action_dim * reward_dim,
            net_arch=net_arch[1:],
        )
        self.apply(layer_init)

    def forward(self, obs):
        """Predict Q values for all actions."""
        input = self.feature_extractor(obs)
        q_values = self.net(input)
        return q_values.view(-1, self.action_dim, self.reward_dim)  # Batch size X Actions X Rewards


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class MOSACDiscreteActor(nn.Module):
    """Actor network: S -> A. Does not need any multi-objective concept."""

    def __init__(
        self,
        obs_shape: Tuple,
        action_dim: int,
        reward_dim: int,
        net_arch=[256, 256],
    ):
        """Initialize SAC actor."""
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.net_arch = net_arch

        if len(obs_shape) == 1:
            self.feature_extractor = mlp(obs_shape[0], -1, net_arch[:1])
        elif len(obs_shape) > 1:  # Image observation
            self.feature_extractor = NatureCNN(self.obs_shape, features_dim=net_arch[0])

        self.net = mlp(net_arch[0], action_dim, net_arch[1:])
        self.apply(layer_init)

    def forward(self, x):
        """Forward pass of the actor network."""
        input = self.feature_extractor(x)
        logits = self.net(input)

        return logits

    def get_action(self, x):
        """Get action from the actor network."""
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


class MOSACDiscrete(MOPolicy):
    """Multi-objective Soft Actor-Critic (SAC) algorithm for discrete action spaces.

    It is a multi-objective version of the SAC algorithm, with multi-objective critic and weighted sum scalarization.
    """

    def __init__(
        self,
        env: gym.Env,
        weights: np.ndarray,
        scalarization=th.matmul,
        buffer_size: int = int(1e6),
        gamma: float = 0.99,
        tau: float = 1.0,
        batch_size: int = 128,
        learning_starts: int = int(2e4),
        net_arch=[256, 256],
        policy_lr: float = 3e-4,
        q_lr: float = 3e-4,
        update_frequency: int = 4,
        target_net_freq: int = 2000,
        alpha: float = 0.2,
        autotune: bool = True,
        target_entropy_scale: float = 0.89,
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
            update_frequency: frequency of training updates
            target_net_freq: the frequency of updates for the target networks
            alpha: Entropy regularization coefficient
            autotune: automatic tuning of alpha
            target_entropy_scale: coefficient for scaling the autotune entropy target
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
        assert isinstance(self.env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
        self.obs_shape = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n
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
        self.update_frequency = update_frequency
        self.target_net_freq = target_net_freq
        assert self.target_net_freq % self.update_frequency == 0, "target_net_freq should be divisible by update_frequency"
        self.target_entropy_scale = target_entropy_scale

        # Networks
        self.actor = MOSACDiscreteActor(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            net_arch=self.net_arch,
        ).to(self.device)

        self.qf1 = MODiscreteSoftQNetwork(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            net_arch=self.net_arch,
        ).to(self.device)
        self.qf2 = MODiscreteSoftQNetwork(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            net_arch=self.net_arch,
        ).to(self.device)
        self.qf1_target = MODiscreteSoftQNetwork(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            net_arch=self.net_arch,
        ).to(self.device)
        self.qf2_target = MODiscreteSoftQNetwork(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            net_arch=self.net_arch,
        ).to(self.device)
        self.qf1_target.requires_grad_(False)
        self.qf2_target.requires_grad_(False)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=self.q_lr,
            eps=1e-4,
        )
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.policy_lr, eps=1e-4)

        # Automatic entropy tuning
        self.autotune = autotune
        if self.autotune:
            self.target_entropy = -self.target_entropy_scale * th.log(1 / th.tensor(self.action_dim))
            self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr, eps=1e-4)
        else:
            self.alpha = alpha
        self.alpha_tensor = th.scalar_tensor(self.alpha).to(self.device)

        # Buffer
        self.env.observation_space.dtype = np.float32
        self.buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            action_dim=1,  # output singular index for action
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
            "update_frequency": self.update_frequency,
            "target_net_freq": self.target_net_freq,
            "alpha": self.alpha,
            "autotune": self.autotune,
            "target_entropy_scale": self.target_entropy_scale,
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
            update_frequency=self.update_frequency,
            target_net_freq=self.target_net_freq,
            alpha=self.alpha,
            autotune=self.autotune,
            target_entropy_scale=self.target_entropy_scale,
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
        copied.actor_optimizer = optim.Adam(copied.actor.parameters(), lr=self.policy_lr, eps=1e-4)
        copied.q_optimizer = optim.Adam(
            list(copied.qf1.parameters()) + list(copied.qf2.parameters()),
            lr=self.q_lr,
            eps=1e-4,
        )
        if self.autotune:
            copied.a_optimizer = optim.Adam([copied.log_alpha], lr=self.q_lr, eps=1e-4)
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

        if self.autotune:
            save_dict["log_alpha"] = self.log_alpha
            save_dict["a_optimizer_state_dict"] = self.a_optimizer.state_dict()
            save_dict["target_entropy_scale"] = self.target_entropy_scale

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

        if "log_alpha" in save_dict:  # previously used autotune
            self.log_alpha = save_dict["log_alpha"]
            self.a_optimizer.load_state_dict(save_dict["a_optimizer_state_dict"])
            self.target_entropy_scale = save_dict["target_entropy_scale"]

        if load_replay_buffer:
            self.buffer = save_dict["buffer"]

        self.weights = save_dict["weights"]
        self.alpha = save_dict["alpha"]

    @override
    def eval(
        self,
        obs: np.ndarray,
        w: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Union[int, np.ndarray]:
        """Returns the best action to perform for the given obs.

        Args:
            obs: observation as a numpy array
            w: None
        Return:
            action as a numpy array (discrete actions)
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
        actor_obs = mb_obs
        actor_next_obs = mb_next_obs

        with th.no_grad():
            _, next_state_log_pi, next_state_action_probs = self.actor.get_action(actor_next_obs)
            # (!) Q values are scalarized before being compared (min of ensemble networks)
            qf1_next_target = self.scalarization(self.qf1_target(mb_next_obs), self.weights_tensor)  # (B, A, R) -> (B, A)
            qf2_next_target = self.scalarization(self.qf2_target(mb_next_obs), self.weights_tensor)
            # we can use the action probabilities instead of MC sampling to estimate the expectation
            min_qf_next_target = next_state_action_probs * (
                th.min(qf1_next_target, qf2_next_target) - self.alpha_tensor * next_state_log_pi
            )
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=1)
            scalarized_rewards = self.scalarization(mb_rewards, self.weights_tensor)
            next_q_value = scalarized_rewards.flatten() + (1 - mb_dones.flatten()) * self.gamma * (min_qf_next_target)

        qf1_values = self.scalarization(self.qf1(mb_obs), self.weights_tensor)  # (B, A, R) -> (B, A)
        qf2_values = self.scalarization(self.qf2(mb_obs), self.weights_tensor)
        qf1_a_values = qf1_values.gather(1, mb_act.long()).view(-1)
        qf2_a_values = qf2_values.gather(1, mb_act.long()).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        self.q_optimizer.step()

        _, log_pi, action_probs = self.actor.get_action(actor_obs)
        with th.no_grad():
            # (!) Q values are scalarized before being compared (min of ensemble networks)
            qf1_values = self.scalarization(self.qf1(mb_obs), self.weights_tensor)
            qf2_values = self.scalarization(self.qf2(mb_obs), self.weights_tensor)
            min_qf_values = th.min(qf1_values, qf2_values)
        actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.autotune:
            # reuse action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())).mean()

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

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        start_time=None,
        verbose: bool = False,
    ):
        """Train the agent.

        Args:
            total_timesteps (int): Total number of timesteps (env steps) to train for
            eval_env (Optional[gym.Env]): Gym environment used for evaluation.
            start_time (Optional[float]): Starting time for the training procedure. If None, it will be set to the current time.
            verbose (bool): whether to print the episode info.
        """
        if start_time is None:
            start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = self.env.reset()
        for _ in range(total_timesteps):
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
                        verbose=verbose,
                    )

            # ALGO LOGIC: training.
            if self.global_step > self.learning_starts:
                if self.global_step % self.update_frequency == 0:
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

"""GPI-PD algorithm with continuous actions."""

import os
import random
from itertools import chain
from typing import List, Optional, Union

import gymnasium
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.evaluation import (
    log_all_multi_policy_metrics,
    log_episode_info,
    policy_evaluation_mo,
)
from morl_baselines.common.model_based.probabilistic_ensemble import (
    ProbabilisticEnsemble,
)
from morl_baselines.common.model_based.utils import ModelEnv, visualize_eval
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.networks import layer_init, mlp, polyak_update
from morl_baselines.common.prioritized_buffer import PrioritizedReplayBuffer
from morl_baselines.common.utils import unique_tol
from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport


class Policy(nn.Module):
    """Policy network."""

    def __init__(self, obs_dim, rew_dim, output_dim, action_space, net_arch=[256, 256]):
        """Initialize the policy network."""
        super().__init__()
        self.action_space = action_space
        self.latent_pi = mlp(obs_dim + rew_dim, -1, net_arch)
        self.mean = nn.Linear(net_arch[-1], output_dim)

        # action rescaling
        self.register_buffer("action_scale", th.tensor((action_space.high - action_space.low) / 2.0, dtype=th.float32))
        self.register_buffer("action_bias", th.tensor((action_space.high + action_space.low) / 2.0, dtype=th.float32))

        self.apply(layer_init)

    def forward(self, obs, w, noise=None, noise_clip=None):
        """Forward pass of the policy network."""
        h = self.latent_pi(th.concat((obs, w), dim=obs.dim() - 1))
        action = self.mean(h)
        action = th.tanh(action)
        if noise is not None:
            n = (th.randn_like(action) * noise).clamp(-noise_clip, noise_clip)
            action = (action + n).clamp(-1, 1)
        return action * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    """Q-network S x Ax W -> R^reward_dim."""

    def __init__(self, obs_dim, action_dim, rew_dim, net_arch=[256, 256], layer_norm=True, drop_rate=0.01):
        """Initialize the Q-network."""
        super().__init__()
        self.net = mlp(obs_dim + action_dim + rew_dim, rew_dim, net_arch, drop_rate=drop_rate, layer_norm=layer_norm)
        self.apply(layer_init)

    def forward(self, obs, action, w):
        """Forward pass of the Q-network."""
        q_values = self.net(th.cat((obs, action, w), dim=obs.dim() - 1))
        return q_values


class GPIPDContinuousAction(MOAgent, MOPolicy):
    """GPI-PD algorithm with continuous actions.

    Sample-Efficient Multi-Objective Learning via Generalized Policy Improvement Prioritization
    Lucas N. Alegre, Ana L. C. Bazzan, Diederik M. Roijers, Ann Nowé, Bruno C. da Silva
    AAMAS 2023
    Paper: https://arxiv.org/abs/2301.07784
    See Appendix for Continuous Action details.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 400000,
        net_arch: List = [256, 256],
        batch_size: int = 128,
        num_q_nets: int = 2,
        delay_policy_update: int = 2,
        learning_starts: int = 100,
        gradient_updates: int = 20,
        use_gpi: bool = False,  # In the continuous action case, GPI is only used to selected weights.
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        per: bool = True,
        min_priority: float = 0.1,
        alpha: float = 0.6,
        dyna: bool = True,
        dynamics_net_arch: List = [200, 200, 200, 200],
        dynamics_train_freq: int = 250,
        dynamics_rollout_len: int = 5,
        dynamics_rollout_starts: int = 1000,
        dynamics_rollout_freq: int = 250,
        dynamics_rollout_batch_size: int = 50000,
        dynamics_buffer_size: int = 200000,
        dynamics_min_uncertainty: float = 2.0,
        dynamics_real_ratio: float = 0.1,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "GPI-PD Continuous Action",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
    ):
        """GPI-PD algorithm with continuous actions.

        It extends the TD3 algorithm to multi-objective RL.
        It learns the policy and Q-networks conditioned on the weight vector.

        Args:
            env (gym.Env): The environment to train on.
            learning_rate (float, optional): The learning rate. Defaults to 3e-4.
            gamma (float, optional): The discount factor. Defaults to 0.99.
            tau (float, optional): The soft update coefficient. Defaults to 0.005.
            buffer_size (int, optional): The size of the replay buffer. Defaults to int(1e6).
            net_arch (List, optional): The network architecture for the policy and Q-networks.
            dynamics_net_arch (List, optional): The network architecture for the dynamics model.
            batch_size (int, optional): The batch size for training. Defaults to 256.
            num_q_nets (int, optional): The number of Q-networks to use. Defaults to 2.
            delay_policy_update (int, optional): The number of gradient steps to take before updating the policy. Defaults to 2.
            learning_starts (int, optional): The number of steps to take before starting to train. Defaults to 100.
            gradient_updates (int, optional): The number of gradient steps to take per update. Defaults to 1.
            use_gpi (bool, optional): Whether to use GPI for selecting actions. Defaults to True.
            policy_noise (float, optional): The noise to add to the policy. Defaults to 0.2.
            noise_clip (float, optional): The noise clipping value. Defaults to 0.5.
            per (bool, optional): Whether to use prioritized experience replay. Defaults to False.
            min_priority (float, optional): The minimum priority to use for prioritized experience replay. Defaults to 0.1.
            alpha (float, optional): The alpha value for prioritized experience replay. Defaults to 0.6.
            dyna (bool, optional): Whether to use Dyna. Defaults to False.
            dynamics_train_freq (int, optional): The frequency with which to train the dynamics model. Defaults to 1000.
            dynamics_rollout_len (int, optional): The rollout length for the dynamics model. Defaults to 1.
            dynamics_rollout_starts (int, optional): The number of steps to take before starting to train the dynamics model. Defaults to 5000.
            dynamics_rollout_freq (int, optional): The frequency with which to rollout the dynamics model. Defaults to 250.
            dynamics_rollout_batch_size (int, optional): The batch size for the dynamics model rollout. Defaults to 10000.
            dynamics_buffer_size (int, optional): The size of the dynamics model replay buffer. Defaults to 400000.
            dynamics_min_uncertainty (float, optional): The minimum uncertainty to use for the dynamics model. Defaults to 1.0.
            dynamics_real_ratio (float, optional): The ratio of real data to use for the dynamics model. Defaults to 0.1.
            project_name (str, optional): The name of the project. Defaults to "MORL Baselines".
            experiment_name (str, optional): The name of the experiment. Defaults to "GPI-PD Continuous Action".
            wandb_entity (Optional[str], optional): The wandb entity. Defaults to None.
            log (bool, optional): Whether to log to wandb. Defaults to True.
            seed (Optional[int], optional): The seed to use. Defaults to None.
            device (Union[th.device, str], optional): The device to use for training. Defaults to "auto".
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device=device)
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.use_gpi = use_gpi
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.buffer_size = buffer_size
        self.num_q_nets = num_q_nets
        self.delay_policy_update = delay_policy_update
        self.net_arch = net_arch
        self.dynamics_net_arch = dynamics_net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates
        self.per = per
        self.min_priority = min_priority
        self.alpha = alpha
        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.observation_shape, self.action_dim, rew_dim=self.reward_dim, max_size=buffer_size
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.observation_shape, self.action_dim, rew_dim=self.reward_dim, max_size=buffer_size
            )

        self.q_nets = [
            QNetwork(self.observation_dim, self.action_dim, self.reward_dim, net_arch=net_arch).to(self.device)
            for _ in range(num_q_nets)
        ]
        self.target_q_nets = [
            QNetwork(self.observation_dim, self.action_dim, self.reward_dim, net_arch=net_arch).to(self.device)
            for _ in range(num_q_nets)
        ]
        for q_net, target_q_net in zip(self.q_nets, self.target_q_nets):
            target_q_net.load_state_dict(q_net.state_dict())
            for param in target_q_net.parameters():
                param.requires_grad = False

        self.policy = Policy(
            self.observation_dim, self.reward_dim, self.action_dim, self.env.action_space, net_arch=net_arch
        ).to(self.device)
        self.target_policy = Policy(
            self.observation_dim, self.reward_dim, self.action_dim, self.env.action_space, net_arch=net_arch
        ).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        for param in self.target_policy.parameters():
            param.requires_grad = False

        self.q_optim = optim.Adam(chain(*[net.parameters() for net in self.q_nets]), lr=self.learning_rate)
        self.policy_optim = optim.Adam(list(self.policy.parameters()), lr=self.learning_rate)

        self.dyna = dyna
        self.dynamics = None
        self.dynamics_buffer = None
        if self.dyna:
            self.dynamics = ProbabilisticEnsemble(
                input_dim=self.observation_dim + self.action_dim,
                output_dim=self.observation_dim + self.reward_dim,
                arch=self.dynamics_net_arch,
                device=self.device,
            )
            self.dynamics_buffer = ReplayBuffer(
                self.observation_shape, self.action_dim, rew_dim=self.reward_dim, max_size=dynamics_buffer_size
            )
        self.dynamics_train_freq = dynamics_train_freq
        self.dynamics_rollout_len = dynamics_rollout_len
        self.dynamics_rollout_starts = dynamics_rollout_starts
        self.dynamics_rollout_freq = dynamics_rollout_freq
        self.dynamics_rollout_batch_size = dynamics_rollout_batch_size
        self.dynamics_min_uncertainty = dynamics_min_uncertainty
        self.dynamics_real_ratio = dynamics_real_ratio

        self.weight_support = []
        self.stacked_weight_support = []

        self._n_updates = 0

        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def get_config(self):
        """Get the configuration of the agent."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "num_q_nets": self.num_q_nets,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "policy_noise": self.policy_noise,
            "net_arch": self.net_arch,
            "gradient_updates": self.gradient_updates,
            "delay_policy_update": self.delay_policy_update,
            "min_priority": self.min_priority,
            "per": self.per,
            "buffer_size": self.buffer_size,
            "alpha": self.alpha,
            "learning_starts": self.learning_starts,
            "dyna": self.dyna,
            "dynamics_net_arch": self.dynamics_net_arch,
            "dynamics_rollout_len": self.dynamics_rollout_len,
            "dynamics_min_uncertainty": self.dynamics_min_uncertainty,
            "dynamics_real_ratio": self.dynamics_real_ratio,
            "dynamics_train_freq": self.dynamics_train_freq,
            "dynamics_rollout_starts": self.dynamics_rollout_starts,
            "dynamics_rollout_freq": self.dynamics_rollout_freq,
            "dynamics_rollout_batch_size": self.dynamics_rollout_batch_size,
            "seed": self.seed,
        }

    def save(self, save_dir="weights/", filename=None, save_replay_buffer=True):
        """Save the agent's weights and replay buffer."""
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        saved_params = {
            "policy_state_dict": self.policy.state_dict(),
            "policy_optimizer_state_dict": self.policy_optim.state_dict(),
        }
        for i, (q_net, target_q_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            saved_params["q_net_" + str(i) + "_state_dict"] = q_net.state_dict()
            saved_params["target_q_net_" + str(i) + "_state_dict"] = target_q_net.state_dict()
        saved_params["q_nets_optimizer_state_dict"] = self.q_optim.state_dict()
        saved_params["M"] = self.weight_support
        if self.dyna:
            saved_params["dynamics_state_dict"] = self.dynamics.state_dict()
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = self.experiment_name if filename is None else filename
        th.save(saved_params, save_dir + "/" + filename + ".tar")

    def load(self, path, load_replay_buffer=True):
        """Load the agent weights from a file."""
        params = th.load(path, map_location=self.device, weights_only=False)
        self.weight_support = params["M"]
        self.stacked_weight_support = th.stack(self.weight_support)
        self.policy.load_state_dict(params["policy_state_dict"])
        self.policy_optim.load_state_dict(params["policy_optimizer_state_dict"])
        for i, (q_net, target_q_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            q_net.load_state_dict(params["q_net_" + str(i) + "_state_dict"])
            target_q_net.load_state_dict(params["target_q_net_" + str(i) + "_state_dict"])
        self.q_optim.load_state_dict(params["q_nets_optimizer_state_dict"])
        if self.dyna:
            self.dynamics.load_state_dict(params["dynamics_state_dict"])
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def _sample_batch_experiences(self):
        if not self.dyna or self.global_step < self.dynamics_rollout_starts or len(self.dynamics_buffer) == 0:
            return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)
        else:
            num_real_samples = int(self.batch_size * self.dynamics_real_ratio)  # % of real world data
            if self.per:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones, idxes = self.replay_buffer.sample(
                    num_real_samples, to_tensor=True, device=self.device
                )
            else:
                (s_obs, s_actions, s_rewards, s_next_obs, s_dones) = self.replay_buffer.sample(
                    num_real_samples, to_tensor=True, device=self.device
                )
            (m_obs, m_actions, m_rewards, m_next_obs, m_dones) = self.dynamics_buffer.sample(
                self.batch_size - num_real_samples, to_tensor=True, device=self.device
            )
            experience_tuples = (
                th.cat([s_obs, m_obs], dim=0),
                th.cat([s_actions, m_actions], dim=0),
                th.cat([s_rewards, m_rewards], dim=0),
                th.cat([s_next_obs, m_next_obs], dim=0),
                th.cat([s_dones, m_dones], dim=0),
            )
            if self.per:
                return experience_tuples + (idxes,)
            return experience_tuples

    @th.no_grad()
    def _rollout_dynamics(self, weight: th.Tensor):
        # Dyna Planning
        num_times = int(np.ceil(self.dynamics_rollout_batch_size / 10000))
        batch_size = min(self.dynamics_rollout_batch_size, 10000)
        for _ in range(num_times):
            obs = self.replay_buffer.sample_obs(batch_size, to_tensor=False)
            model_env = ModelEnv(self.dynamics, self.env.unwrapped.spec.id, rew_dim=self.reward_dim)
            for plan_step in range(self.dynamics_rollout_len):
                obs = th.tensor(obs).to(self.device)
                w = weight.repeat(obs.shape[0], 1)
                actions = self.policy(obs, w, noise=self.policy_noise, noise_clip=self.noise_clip)

                next_obs_pred, r_pred, dones, info = model_env.step(obs, actions)
                obs, actions = (obs.detach().cpu().numpy(), actions.detach().cpu().numpy())

                uncertainties = info["uncertainty"]
                for i in range(len(obs)):
                    if uncertainties[i] < self.dynamics_min_uncertainty:
                        self.dynamics_buffer.add(obs[i], actions[i], r_pred[i], next_obs_pred[i], dones[i])

                nonterm_mask = ~dones.squeeze(-1)
                if nonterm_mask.sum() == 0:
                    break

                obs = next_obs_pred[nonterm_mask]

        if self.log:
            wandb.log(
                {
                    "dynamics/uncertainty_mean": uncertainties.mean(),
                    "dynamics/uncertainty_max": uncertainties.max(),
                    "dynamics/uncertainty_min": uncertainties.min(),
                    "global_step": self.global_step,
                },
            )

    def update(self, weight: th.Tensor):
        """Update the policy and the Q-nets."""
        for _ in range(self.gradient_updates):
            if self.per:
                (s_obs, s_actions, s_rewards, s_next_obs, s_dones, idxes) = self._sample_batch_experiences()
            else:
                (s_obs, s_actions, s_rewards, s_next_obs, s_dones) = self._sample_batch_experiences()

            if len(self.weight_support) > 1:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = (
                    s_obs.repeat(2, 1),
                    s_actions.repeat(2, 1),
                    s_rewards.repeat(2, 1),
                    s_next_obs.repeat(2, 1),
                    s_dones.repeat(2, 1),
                )
                w = th.vstack(
                    [weight for _ in range(s_obs.size(0) // 2)] + random.choices(self.weight_support, k=s_obs.size(0) // 2)
                )
            else:
                w = weight.repeat(s_obs.size(0), 1)

            with th.no_grad():
                next_actions = self.target_policy(s_next_obs, w, noise=self.policy_noise, noise_clip=self.noise_clip)
                q_targets = th.stack([q_target(s_next_obs, next_actions, w) for q_target in self.target_q_nets])
                scalarized_q_targets = th.einsum("nbr,br->nb", q_targets, w)
                inds = th.argmin(scalarized_q_targets, dim=0, keepdim=True)
                inds = inds.reshape(1, -1, 1).expand(1, q_targets.size(1), q_targets.size(2))
                target_q = q_targets.gather(0, inds).squeeze(0)

                target_q = (s_rewards + (1 - s_dones) * self.gamma * target_q).detach()

            q_values = [q_net(s_obs, s_actions, w) for q_net in self.q_nets]
            critic_loss = (1 / self.num_q_nets) * sum([F.mse_loss(q_value, target_q) for q_value in q_values])

            self.q_optim.zero_grad()
            critic_loss.backward()
            self.q_optim.step()

            if self.per:
                per = (q_values[0] - target_q)[: len(idxes)].detach().abs() * 0.05
                per = th.einsum("br,br->b", per, w[: len(idxes)])
                priority = per.cpu().numpy().flatten()
                priority = priority.clip(min=self.min_priority) ** self.alpha
                self.replay_buffer.update_priorities(idxes, priority)

            for q_net, target_q_net in zip(self.q_nets, self.target_q_nets):
                polyak_update(q_net.parameters(), target_q_net.parameters(), self.tau)

            if self._n_updates % self.delay_policy_update == 0:
                # Policy update
                actions = self.policy(s_obs, w)
                q_values_pi = (1 / self.num_q_nets) * sum(q_net(s_obs, actions, w) for q_net in self.q_nets)
                policy_loss = -th.einsum("br,br->b", q_values_pi, w).mean()

                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()

                polyak_update(self.policy.parameters(), self.target_policy.parameters(), self.tau)

            self._n_updates += 1

        if self.log and self.global_step % 100 == 0:
            if self.per:
                wandb.log(
                    {
                        "metrics/mean_priority": np.mean(priority),
                        "metrics/max_priority": np.max(priority),
                        "metrics/min_priority": np.min(priority),
                    },
                    commit=False,
                )
            wandb.log(
                {
                    "losses/critic_loss": critic_loss.item(),
                    "losses/policy_loss": policy_loss.item(),
                    "global_step": self.global_step,
                },
            )

    @th.no_grad()
    def eval(
        self, obs: Union[np.ndarray, th.Tensor], w: Union[np.ndarray, th.Tensor], torch_action=False
    ) -> Union[np.ndarray, th.Tensor]:
        """Evaluate the policy action for the given observation and weight vector."""
        if isinstance(obs, np.ndarray):
            obs = th.tensor(obs).float().to(self.device)
            w = th.tensor(w).float().to(self.device)

        if self.use_gpi:
            obs = obs.repeat(len(self.weight_support), 1)
            actions_original = self.policy(obs, self.stacked_weight_support)

            obs = obs.repeat(len(self.weight_support), 1, 1)
            actions = actions_original.repeat(len(self.weight_support), 1, 1)
            stackedM = self.stacked_weight_support.repeat_interleave(len(self.weight_support), dim=0).view(
                len(self.weight_support), len(self.weight_support), self.reward_dim
            )
            values = self.q_nets[0](obs, actions, stackedM)

            scalar_values = th.einsum("par,r->pa", values, w)
            max_q, a = th.max(scalar_values, dim=1)
            policy_index = th.argmax(max_q)  # max_i max_a q(s,a,w_i)
            action = a[policy_index].detach().item()
            action = actions_original[action]
        else:
            action = self.policy(obs, w)

        if not torch_action:
            action = action.detach().cpu().numpy()

        return action

    def set_weight_support(self, weight_list: List[np.ndarray]):
        """Set the weight support set."""
        weights_no_repeat = unique_tol(weight_list)
        self.weight_support = [th.tensor(w).float().to(self.device) for w in weights_no_repeat]
        if len(self.weight_support) > 0:
            self.stacked_weight_support = th.stack(self.weight_support)

    def train_iteration(
        self,
        total_timesteps: int,
        weight: np.ndarray,
        weight_support: List[np.ndarray],
        change_weight_every_episode: bool = False,
        eval_env=None,
        eval_freq: int = 1000,
        reset_num_timesteps: bool = False,
    ):
        """Train the agent.

        Args:
            total_timesteps (int): Total number of timesteps to train the agent for.
            weight (np.ndarray): Initial weight vector.
            weight_support (List[np.ndarray]): List of weight vectors to use for the weight support set.
            change_weight_every_episode (bool): Whether to change the weight vector at the end of each episode.
            eval_env (Optional[gym.Env]): Environment to use for evaluation.
            eval_freq (int): Number of timesteps between evaluations.
            reset_num_timesteps (bool): Whether to reset the number of timesteps.
        """
        weight_support = unique_tol(weight_support)
        self.set_weight_support(weight_support)
        tensor_w = th.tensor(weight).float().to(self.device)

        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes

        obs, info = self.env.reset()
        for _ in range(1, total_timesteps + 1):
            self.global_step += 1

            if self.global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                with th.no_grad():
                    action = (
                        self.policy(
                            th.tensor(obs).float().to(self.device),
                            tensor_w,
                            noise=self.policy_noise,
                            noise_clip=self.noise_clip,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

            action_env = action

            next_obs, vector_reward, terminated, truncated, info = self.env.step(action_env)

            self.replay_buffer.add(obs, action, vector_reward, next_obs, terminated)

            if self.global_step >= self.learning_starts:
                if self.dyna:
                    if self.global_step % self.dynamics_train_freq == 0:
                        (m_obs, m_actions, m_rewards, m_next_obs, m_dones) = self.replay_buffer.get_all_data()
                        X = np.hstack((m_obs, m_actions))
                        Y = np.hstack((m_rewards, m_next_obs - m_obs))
                        mean_holdout_loss = self.dynamics.fit(X, Y)
                        if self.log:
                            wandb.log(
                                {"dynamics/mean_holdout_loss": mean_holdout_loss, "global_step": self.global_step},
                            )

                    if self.global_step >= self.dynamics_rollout_starts and self.global_step % self.dynamics_rollout_freq == 0:
                        self._rollout_dynamics(tensor_w)

                self.update(tensor_w)

            if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                self.policy_eval(eval_env, weights=weight, log=self.log)

                if self.dyna and self.global_step >= self.dynamics_rollout_starts:
                    plot = visualize_eval(self, eval_env, self.dynamics, w=weight, compound=False, horizon=1000)
                    wandb.log({"dynamics/predictions": wandb.Image(plot), "global_step": self.global_step})
                    plot.close()

            if terminated or truncated:
                obs, _ = self.env.reset()
                self.num_episodes += 1

                if self.log and "episode" in info.keys():
                    log_episode_info(info["episode"], np.dot, weight, self.global_step)

                if change_weight_every_episode:
                    weight = random.choice(weight_support)
                    tensor_w = th.tensor(weight).float().to(self.device)
            else:
                obs = next_obs

    def train(
        self,
        total_timesteps: int,
        eval_env: gymnasium.Env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        num_eval_weights_for_eval: int = 50,
        weight_selection_algo: str = "gpi-ls",
        timesteps_per_iter: int = 10000,
        eval_freq: int = 1000,
        eval_mo_freq: int = 10000,
        checkpoints: bool = True,
    ):
        """Train the agent.

        Args:
            total_timesteps (int): Total number of timesteps to train the agent for.
            eval_env (gym.Env): Environment to use for evaluation.
            ref_point (np.ndarray): Reference point for hypervolume calculation.
            known_pareto_front (Optional[List[np.ndarray]]): Optimal Pareto front, if known.
            num_eval_weights_for_front (int): Number of weights to evaluate for the Pareto front.
            num_eval_episodes_for_front: number of episodes to run when evaluating the policy.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            weight_selection_algo (str): Weight selection algorithm to use.
            timesteps_per_iter (int): Number of timesteps to train the agent for each iteration.
            eval_freq (int): Number of timesteps between evaluations during an iteration.
            eval_mo_freq (int): Number of timesteps between multi-objective evaluations.
            checkpoints (bool): Whether to save checkpoints.
        """
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_front": num_eval_weights_for_front,
                    "num_eval_episodes_for_front": num_eval_episodes_for_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "weight_selection_algo": weight_selection_algo,
                    "timesteps_per_iter": timesteps_per_iter,
                    "eval_freq": eval_freq,
                    "eval_mo_freq": eval_mo_freq,
                }
            )
        max_iter = total_timesteps // timesteps_per_iter
        linear_support = LinearSupport(num_objectives=self.reward_dim, epsilon=0.0 if weight_selection_algo == "ols" else None)

        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)

        for iter in range(1, max_iter + 1):
            if weight_selection_algo == "ols" or weight_selection_algo == "gpi-ls":
                if weight_selection_algo == "gpi-ls":
                    self.set_weight_support(linear_support.get_weight_support())
                    use_gpi = self.use_gpi
                    self.use_gpi = True
                    w = linear_support.next_weight(
                        algo="gpi-ls", gpi_agent=self, env=eval_env, rep_eval=num_eval_episodes_for_front
                    )
                    self.use_gpi = use_gpi
                else:
                    w = linear_support.next_weight(algo="ols")

                if w is None:
                    break
            else:
                raise ValueError(f"Unknown algorithm {weight_selection_algo}.")

            print("Next weight vector:", w)
            if weight_selection_algo == "gpi-ls":
                M = linear_support.get_weight_support() + linear_support.get_corner_weights(top_k=4) + [w]
            elif weight_selection_algo == "ols":
                M = linear_support.get_weight_support() + [w]
            else:
                M = None

            self.train_iteration(
                total_timesteps=timesteps_per_iter,
                weight=w,
                weight_support=M,
                change_weight_every_episode=weight_selection_algo == "gpi-ls",
                eval_env=eval_env,
                eval_freq=eval_freq,
            )

            if weight_selection_algo == "ols":
                value = policy_evaluation_mo(self, eval_env, w, rep=num_eval_episodes_for_front)[3]
                linear_support.add_solution(value, w)
            elif weight_selection_algo == "gpi-ls":
                for wcw in M:
                    n_value = policy_evaluation_mo(self, eval_env, wcw, rep=num_eval_episodes_for_front)[3]
                    linear_support.add_solution(n_value, wcw)

            if self.log and self.global_step % eval_mo_freq == 0:
                # Evaluation
                gpi_returns_test_tasks = [
                    policy_evaluation_mo(self, eval_env, ew, rep=num_eval_episodes_for_front)[3] for ew in eval_weights
                ]
                log_all_multi_policy_metrics(
                    current_front=gpi_returns_test_tasks,
                    hv_ref_point=ref_point,
                    reward_dim=self.reward_dim,
                    global_step=self.global_step,
                    n_sample_weights=num_eval_weights_for_eval,
                    ref_front=known_pareto_front,
                )
                # This is the EU computed in the paper
                mean_gpi_returns_test_tasks = np.mean(
                    [np.dot(ew, q) for ew, q in zip(eval_weights, gpi_returns_test_tasks)], axis=0
                )
                wandb.log({"eval/Mean Utility - GPI": mean_gpi_returns_test_tasks, "iteration": iter})

            # Checkpoint
            if checkpoints:
                self.save(filename=f"GPI-PD {weight_selection_algo} iter={iter}", save_replay_buffer=False)

        self.close_wandb()


class GPILSContinuousAction(GPIPDContinuousAction):
    """Model-free version of GPI-PD with continuous actions."""

    def __init__(self, *args, **kwargs):
        """Initialize the agent deactivating the dynamics model."""
        super().__init__(dyna=False, experiment_name="GPI-LS Continuous Action", *args, **kwargs)

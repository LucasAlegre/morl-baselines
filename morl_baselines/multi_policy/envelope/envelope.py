import os
import random
from typing import Callable, List, Optional, Union

import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb as wb
from morl_baselines.common.morl_algorithm import MOPolicy, MOAgent
from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.prioritized_buffer import PrioritizedReplayBuffer
from morl_baselines.common.networks import mlp, NatureCNN
from morl_baselines.common.utils import (
    layer_init,
    polyak_update,
    linearly_decaying_value,
    get_grad_norm,
    huber,
    random_weights,
    log_episode_info,
)
from mo_gym.evaluation import eval_mo


class QNet(nn.Module):
    def __init__(self, obs_shape, action_dim, rew_dim, net_arch):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.rew_dim = rew_dim
        if len(obs_shape) == 1:
            self.feature_extractor = None
            input_dim = obs_shape[0] + rew_dim
        elif len(obs_shape) > 1:  # Image observation
            self.feature_extractor = NatureCNN(self.obs_shape, features_dim=512)
            input_dim = self.feature_extractor.features_dim + rew_dim
        self.net = mlp(input_dim, action_dim * rew_dim, net_arch)
        self.apply(layer_init)

    def forward(self, obs, w):
        if self.feature_extractor is not None:
            features = self.feature_extractor(obs / 255.0)
            input = th.cat((features, w), dim=w.dim() - 1)
        else:
            input = th.cat((obs, w), dim=w.dim() - 1)
        q_values = self.net(input)
        return q_values.view(-1, self.action_dim, self.rew_dim)  # Batch size X Actions X Rewards


class Envelope(MOPolicy, MOAgent):
    """
    R. Yang, X. Sun, and K. Narasimhan, “A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation,”
    arXiv:1908.08342 [cs], Nov. 2019, Accessed: Sep. 06, 2021. [Online]. Available: http://arxiv.org/abs/1908.08342

    Envelope uses a conditoned network to embed multiple policies (taking the weight as input).

    The main change of this algorithm compare to a scalarized CN DQN is the target update.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        initial_epsilon: float = 0.01,
        final_epsilon: float = 0.01,
        epsilon_decay_steps: int = None,  # None == fixed epsilon
        tau: float = 1.0,
        target_net_update_freq: int = 1000,  # ignored if tau != 1.0
        buffer_size: int = int(1e6),
        net_arch: List = [256, 256],
        batch_size: int = 256,
        learning_starts: int = 100,
        gradient_updates: int = 1,
        gamma: float = 0.99,
        max_grad_norm: Optional[float] = None,
        envelope: bool = True,
        num_sample_w: int = 4,
        per: bool = True,
        per_alpha: float = 0.6,
        initial_homotopy_lambda: float = 0.0,
        final_homotopy_lambda: float = 1.0,
        homotopy_decay_steps: int = None,
        project_name: str = "Envelope",
        experiment_name: str = "Envelope",
        log: bool = True,
        device: Union[th.device, str] = "auto",
    ):

        MOAgent.__init__(self, env, device=device)
        MOPolicy.__init__(self, device)
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.tau = tau
        self.target_net_update_freq = target_net_update_freq
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.per = per
        self.per_alpha = per_alpha
        self.gradient_updates = gradient_updates
        self.initial_homotopy_lambda = initial_homotopy_lambda
        self.final_homotopy_lambda = final_homotopy_lambda
        self.homotopy_decay_steps = homotopy_decay_steps

        self.q_net = QNet(self.observation_shape, self.action_dim, self.reward_dim, net_arch=net_arch).to(self.device)
        self.target_q_net = QNet(self.observation_shape, self.action_dim, self.reward_dim, net_arch=net_arch).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        for param in self.target_q_net.parameters():
            param.requires_grad = False

        self.q_optim = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

        self.envelope = envelope
        self.num_sample_w = num_sample_w
        self.homotopy_lambda = self.initial_homotopy_lambda
        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.observation_shape,
                1,
                rew_dim=self.reward_dim,
                max_size=buffer_size,
                action_dtype=np.uint8,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.observation_shape,
                1,
                rew_dim=self.reward_dim,
                max_size=buffer_size,
                action_dtype=np.uint8,
            )

        self.log = log
        if log:
            self.setup_wandb(project_name, experiment_name)

    def get_config(self):
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps:": self.epsilon_decay_steps,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "clip_grand_norm": self.max_grad_norm,
            "target_net_update_freq": self.target_net_update_freq,
            "gamma": self.gamma,
            "use_envelope": self.envelope,
            "num_sample_w": self.num_sample_w,
            "net_arch": self.net_arch,
            "per": self.per,
            "gradient_updates": self.gradient_updates,
            "buffer_size": self.buffer_size,
            "initial_homotopy_lambda": self.initial_homotopy_lambda,
            "final_homotopy_lambda": self.final_homotopy_lambda,
            "homotopy_decay_steps": self.homotopy_decay_steps,
            "learning_starts": self.learning_starts,
        }

    def save(self, save_replay_buffer=True, save_dir="weights/", filename=None):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        saved_params = {}
        saved_params[f"q_net_state_dict"] = self.q_net.state_dict()

        saved_params["q_net_optimizer_state_dict"] = self.q_optim.state_dict()
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = self.experiment_name if filename is None else filename
        th.save(saved_params, save_dir + "/" + filename + ".tar")

    def load(self, path, load_replay_buffer=True):
        params = th.load(path)
        self.q_net.load_state_dict(params["q_net_state_dict"])
        self.target_q_net.load_state_dict(params["q_net_state_dict"])
        self.q_optim.load_state_dict(params["q_net_optimizer_state_dict"])
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)

    def update(self):
        critic_losses = []
        for g in range(self.gradient_updates):
            if self.per:
                (
                    b_obs,
                    b_actions,
                    b_rewards,
                    b_next_obs,
                    b_dones,
                    b_inds,
                ) = self.sample_batch_experiences()
            else:
                (
                    b_obs,
                    b_actions,
                    b_rewards,
                    b_next_obs,
                    b_dones,
                ) = self.sample_batch_experiences()

            sampled_w = (
                th.tensor(random_weights(dim=self.reward_dim, n=self.num_sample_w)).float().to(self.device)
            )  # sample num_sample_w random weights
            w = sampled_w.repeat_interleave(b_obs.size(0), 0)  # repeat the weights for each sample
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = (
                b_obs.repeat(self.num_sample_w, 1),
                b_actions.repeat(self.num_sample_w, 1),
                b_rewards.repeat(self.num_sample_w, 1),
                b_next_obs.repeat(self.num_sample_w, 1),
                b_dones.repeat(self.num_sample_w, 1),
            )

            with th.no_grad():
                if self.envelope:
                    target = self.envelope_target(b_next_obs, w, sampled_w)
                else:
                    target = self.ddqn_target(b_next_obs, w)
                target_q = b_rewards + (1 - b_dones) * self.gamma * target

            q_values = self.q_net(b_obs, w)
            q_value = q_values.gather(
                1,
                b_actions.long().reshape(-1, 1, 1).expand(q_values.size(0), 1, q_values.size(2)),
            )
            q_value = q_value.reshape(-1, self.reward_dim)

            critic_loss = F.mse_loss(q_value, target_q)

            if self.homotopy_lambda > 0:
                wQ = th.einsum("br,br->b", q_value, w)
                wTQ = th.einsum("br,br->b", target_q, w)
                auxiliary_loss = F.mse_loss(wQ, wTQ)
                critic_loss = (1 - self.homotopy_lambda) * critic_loss + self.homotopy_lambda * auxiliary_loss

            self.q_optim.zero_grad()
            critic_loss.backward()
            if self.log and self.global_step % 100 == 0:
                self.writer.add_scalar(
                    "losses/grad_norm",
                    get_grad_norm(self.q_net.parameters()).item(),
                    self.global_step,
                )
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.q_optim.step()
            critic_losses.append(critic_loss.item())

            if self.per:
                td_err = (q_value[: len(b_inds)] - target_q[: len(b_inds)]).detach()
                priority = th.einsum("sr,sr->s", td_err, w[: len(b_inds)]).abs()
                priority = priority.cpu().numpy().flatten()
                priority = (priority + self.replay_buffer.eps) ** self.per_alpha
                self.replay_buffer.update_priorities(b_inds, priority)

        if self.tau != 1 or self.global_step % self.target_net_update_freq == 0:
            polyak_update(self.q_net.parameters(), self.target_q_net.parameters(), self.tau)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_epsilon,
            )

        if self.homotopy_decay_steps is not None:
            self.homotopy_lambda = linearly_decaying_value(
                self.initial_homotopy_lambda,
                self.homotopy_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_homotopy_lambda,
            )

        if self.log and self.global_step % 100 == 0:
            self.writer.add_scalar("losses/critic_loss", np.mean(critic_losses), self.global_step)
            self.writer.add_scalar("metrics/epsilon", self.epsilon, self.global_step)
            self.writer.add_scalar("metrics/homotopy_lambda", self.homotopy_lambda, self.global_step)

    def eval(self, obs: np.ndarray, w: np.ndarray) -> int:
        obs = th.as_tensor(obs).float().to(self.device)
        w = th.as_tensor(w).float().to(self.device)
        return self.max_action(obs, w)

    def act(self, obs: th.Tensor, w: th.Tensor) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.max_action(obs, w)

    @th.no_grad()
    def max_action(self, obs: th.Tensor, w: th.Tensor) -> int:
        q_values = self.q_net(obs, w)
        scalarized_q_values = th.einsum("r,bar->ba", w, q_values)
        max_act = th.argmax(scalarized_q_values, dim=1)
        return max_act.detach().item()

    @th.no_grad()
    def envelope_target(self, obs: th.Tensor, w: th.Tensor, sampled_w: th.Tensor) -> th.Tensor:
        # Repeat the weights for each sample
        W = sampled_w.unsqueeze(0).repeat(obs.size(0), 1, 1)
        # Repeat the observations for each sampled weight
        next_obs = obs.unsqueeze(1).repeat(1, sampled_w.size(0), 1)

        # Batch size X Num sampled weights X Num actions X Num objectives
        next_q_values = self.q_net(next_obs, W).view(obs.size(0), sampled_w.size(0), self.action_dim, self.reward_dim)
        # Scalarized Q values for each sampled weight
        scalarized_next_q_values = th.einsum("br,bwar->bwa", w, next_q_values)
        # Max Q values for each sampled weight
        max_q, ac = th.max(scalarized_next_q_values, dim=2)
        # Max weights in the envelope
        pref = th.argmax(max_q, dim=1)

        # MO Q-values evaluated on the target network
        next_q_values_target = self.target_q_net(next_obs, W).view(
            obs.size(0), sampled_w.size(0), self.action_dim, self.reward_dim
        )

        # Index the Q-values for the max actions
        max_next_q = next_q_values_target.gather(
            2,
            ac.unsqueeze(2).unsqueeze(3).expand(next_q_values.size(0), next_q_values.size(1), 1, next_q_values.size(3)),
        ).squeeze(2)
        # Index the Q-values for the max sampled weights
        max_next_q = max_next_q.gather(1, pref.reshape(-1, 1, 1).expand(max_next_q.size(0), 1, max_next_q.size(2))).squeeze(1)
        return max_next_q

    @th.no_grad()
    def ddqn_target(self, obs: th.Tensor, w: th.Tensor) -> th.Tensor:
        # Max action for each state
        q_values = self.q_net(obs, w)
        scalarized_q_values = th.einsum("br,bar->ba", w, q_values)
        max_acts = th.argmax(scalarized_q_values, dim=1)
        # Action evaluated with the target network
        q_values_target = self.target_q_net(obs, w)
        q_values_target = q_values_target.gather(
            1,
            max_acts.long().reshape(-1, 1, 1).expand(q_values_target.size(0), 1, q_values_target.size(2)),
        )
        q_values_target = q_values_target.reshape(-1, self.reward_dim)
        return q_values_target

    def train(
        self,
        total_timesteps: int,
        weight: Optional[
            np.ndarray
        ] = None,  # Weight vector. If None, it is randomly sampled every episode (as done in the paper).
        total_episodes: Optional[int] = None,
        reset_num_timesteps: bool = True,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 1000,
        reset_learning_starts: bool = False,
    ):
        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        if reset_learning_starts:  # Resets epsilon-greedy exploration
            self.learning_starts = self.global_step

        num_episodes = 0
        obs, _ = self.env.reset()

        w = weight if weight is not None else random_weights(self.reward_dim, 1)
        tensor_w = th.tensor(w).float().to(self.device)

        for _ in range(1, total_timesteps + 1):
            if total_episodes is not None and num_episodes == total_episodes:
                break
            self.global_step += 1

            if self.global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.act(th.as_tensor(obs).float().to(self.device), tensor_w)

            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)

            self.replay_buffer.add(obs, action, vec_reward, next_obs, terminated)

            if self.global_step >= self.learning_starts:
                self.update()

            if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                self.policy_eval(eval_env, weights=w, writer=self.writer)

            if terminated or truncated:
                obs, _ = self.env.reset()
                num_episodes += 1
                self.num_episodes += 1

                if weight is None:
                    w = random_weights(self.reward_dim, 1)
                    tensor_w = th.tensor(w).float().to(self.device)

                if self.log and "episode" in info.keys():
                    log_episode_info(info["episode"], np.dot, w, self.global_step, self.writer)

            else:
                obs = next_obs

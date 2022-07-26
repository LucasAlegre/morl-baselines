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
from morl_baselines.common.morl_algorithm import MORLAlgorithm
from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.networks import mlp, NatureCNN
from morl_baselines.common.utils import layer_init, polyak_update, linearly_decaying_epsilon, get_grad_norm, huber
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


class Envelope(MORLAlgorithm):
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
        min_priority: float = 1.0,
        project_name: str = "Envelope",
        experiment_name: str = "Envelope",
        log: bool = True,
        device: Union[th.device, str] = "auto",
    ):

        super().__init__(env, device)
        self.rew_dim = len(self.env.w)
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
        self.gradient_updates = gradient_updates

        self.q_net = QNet(self.observation_shape, self.action_dim, self.rew_dim, net_arch=net_arch).to(self.device)
        self.target_q_net = QNet(self.observation_shape, self.action_dim, self.rew_dim, net_arch=net_arch).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        for param in self.target_q_net.parameters():
            param.requires_grad = False

        self.q_optim = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

        self.envelope = envelope
        self.replay_buffer = ReplayBuffer(self.observation_shape, 1, rew_dim=self.rew_dim, max_size=buffer_size, action_dtype=np.uint8)
        self.min_priority = min_priority

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
            "min_priority": self.min_priority,
            "tau": self.tau,
            "clip_grand_norm": self.max_grad_norm,
            "target_net_update_freq": self.target_net_update_freq,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "gradient_updates": self.gradient_updates,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
        }

    def save(self, save_replay_buffer=True, save_dir="weights/", filename=None):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        saved_params = {}
        saved_params[f"q_net_state_dict"] = self.q_net.state_dict()

        saved_params["q_net_optimizer_state_dict"] = self.q_optim.state_dict()
        saved_params["M"] = self.M
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = self.experiment_name if filename is None else filename
        th.save(saved_params, save_dir + "/" + filename + ".tar")

    def load(self, path, load_replay_buffer=True):
        params = th.load(path)
        self.q_net.load_state_dict(params["q_net_state_dict"])
        self.target_q_net.load_state_dict(params["q_net_state_dict"])
        self.q_optim.load_state_dict(params["q_net_optimizer_state_dict"])
        self.M = params["M"]
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)

    def train(self, weight: th.Tensor):
        critic_losses = []
        for g in range(self.gradient_updates):
            s_obs, s_actions, s_rewards, s_next_obs, s_dones = self.sample_batch_experiences()

            if len(self.M) > 1:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = s_obs.repeat(2, 1), s_actions.repeat(2, 1), s_rewards.repeat(2, 1), s_next_obs.repeat(2, 1), s_dones.repeat(2, 1)
                w = th.vstack([weight for _ in range(s_obs.size(0) // 2)] + random.choices(self.M, k=s_obs.size(0) // 2))
            else:
                w = weight.repeat(s_obs.size(0), 1)

            with th.no_grad():
                if self.envelope:
                    target = self.evelope_target(s_next_obs, w)
                else:
                    target = self.ddqn_target(s_next_obs, w)              
                target_q = s_rewards + (1 - s_dones) * self.gamma * target

            q_values = self.q_net(s_obs, w)
            q_value = q_values.gather(1, s_actions.long().reshape(-1, 1, 1).expand(q_values.size(0), 1, q_values.size(2)))
            q_value = q_value.reshape(-1, self.rew_dim)
            td_error = q_value - target_q
            critic_loss = huber(td_error.abs(), min_priority=self.min_priority) 

            self.q_optim.zero_grad()
            critic_loss.backward()
            if self.log and self.num_timesteps % 100 == 0:
                self.writer.add_scalar("losses/grad_norm", get_grad_norm(self.q_net.parameters()).item(), self.num_timesteps)
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.q_optim.step()
            critic_losses.append(critic_loss.item())

        if self.tau != 1 or self.num_timesteps % self.target_net_update_freq == 0:
            polyak_update(self.q_net.parameters(), self.target_q_net.parameters(), self.tau)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_epsilon(self.initial_epsilon, self.epsilon_decay_steps, self.num_timesteps, self.learning_starts, self.final_epsilon)

        if self.log and self.num_timesteps % 100 == 0:
            self.writer.add_scalar("losses/critic_loss", np.mean(critic_losses), self.num_timesteps)
            self.writer.add_scalar("metrics/epsilon", self.epsilon, self.num_timesteps)

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
        scalarized_q_values = th.einsum("r,sar->sa", w, q_values)
        max_act = th.argmax(scalarized_q_values, dim=1)
        return max_act.detach().item()

    @th.no_grad()
    def envelope_target(self, obs: th.Tensor, w: th.Tensor):
        # TODO: There must be a clearer way to write this without all the reshape and expand
        M = th.stack(self.M)
        M = M.unsqueeze(0).repeat(obs.size(0), 1, 1)

        next_obs = obs.unsqueeze(1).repeat(1, M.size(0), 1)
        next_q_values = self.q_net(next_obs, M).view(obs.size(0), len(self.M), self.action_dim, self.rew_dim)
        scalarized_next_q_values = th.einsum("sr,spar->spa", w, next_q_values)
        max_q, ac = th.max(scalarized_next_q_values, dim=2)
        pref = th.argmax(max_q, dim=1)

        next_q_values_target = self.target_q_net(next_obs, M).view(obs.size(0), len(self.M), self.action_dim, self.rew_dim)

        # Max over actions
        max_next_q = next_q_values_target.gather(2, ac.unsqueeze(2).unsqueeze(3).expand(next_q_values.size(0), next_q_values.size(1), 1, next_q_values.size(3))).squeeze(2)
        # Max over preferences
        max_next_q = max_next_q.gather(1, pref.reshape(-1, 1, 1).expand(max_next_q.size(0), 1, max_next_q.size(2))).squeeze(1)
        return max_next_q
    
    @th.no_grad()
    def ddqn_target(self, obs: th.Tensor, w: th.Tensor):
        # Max action for each state
        q_values = self.q_net(obs, w)
        scalarized_q_values = th.einsum("sr,sar->sa", w, q_values)
        max_acts = th.argmax(scalarized_q_values, dim=1)
        # Action evaluated with the target network
        q_values_target = self.target_q_net(obs, w)
        q_values_target = q_values_target.gather(1, max_acts.long().reshape(-1, 1, 1).expand(q_values_target.size(0), 1, q_values_target.size(2)))
        q_values_target = q_values_target.reshape(-1, self.rew_dim)
        return q_values_target

    def learn(
        self,
        total_timesteps: int,
        w: np.ndarray,
        M: List[np.ndarray],
        total_episodes: Optional[int] = None,
        reset_num_timesteps: bool = True,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 1000,
        reset_learning_starts: bool = False,
    ):
        self.env.w = w
        self.M = [th.tensor(w).float().to(self.device) for w in M]
        tensor_w = th.tensor(w).float().to(self.device)

        self.police_indices = []
        self.num_timesteps = 0 if reset_num_timesteps else self.num_timesteps
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        if reset_learning_starts:  # Resets epsilon-greedy exploration
            self.learning_starts = self.num_timesteps

        episode_reward = 0.0
        episode_vec_reward = np.zeros(w.shape[0])
        num_episodes = 0
        obs, done = self.env.reset(), False
        for _ in range(1, total_timesteps + 1):
            if total_episodes is not None and num_episodes == total_episodes:
                break
            self.num_timesteps += 1

            if self.num_timesteps < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.act(th.as_tensor(obs).float().to(self.device), th.as_tensor(w).float().to(self.device))

            next_obs, reward, done, info = self.env.step(action)
            terminal = done if "TimeLimit.truncated" not in info else not info["TimeLimit.truncated"]

            self.replay_buffer.add(obs, action, info["vector_reward"], next_obs, terminal)

            if self.num_timesteps >= self.learning_starts:
                self.train(tensor_w)

            if eval_env is not None and self.log and self.num_timesteps % eval_freq == 0:
                total_reward, discounted_return, total_vec_r, total_vec_return = eval_mo(self, eval_env, w)
                self.writer.add_scalar("eval/total_reward", total_reward, self.num_timesteps)
                self.writer.add_scalar("eval/discounted_return", discounted_return, self.num_timesteps)
                for i in range(episode_vec_reward.shape[0]):
                    self.writer.add_scalar(f"eval/total_reward_obj{i}", total_vec_r[i], self.num_timesteps)
                    self.writer.add_scalar(f"eval/return_obj{i}", total_vec_return[i], self.num_timesteps)

            episode_reward += reward
            episode_vec_reward += info["vector_reward"]
            if done:
                obs, done = self.env.reset(), False
                num_episodes += 1
                self.num_episodes += 1

                if num_episodes % 100 == 0:
                    print(f"Episode: {self.num_episodes} Step: {self.num_timesteps}, Ep. Total Reward: {episode_reward}")
                if self.log:
                    self.police_indices = []
                    self.writer.add_scalar("metrics/episode", self.num_episodes, self.num_timesteps)
                    self.writer.add_scalar("metrics/episode_reward", episode_reward, self.num_timesteps)
                    for i in range(episode_vec_reward.shape[0]):
                        self.writer.add_scalar(f"metrics/episode_reward_obj{i}", episode_vec_reward[i], self.num_timesteps)

                episode_reward = 0.0
                episode_vec_reward = np.zeros(w.shape[0])
            else:
                obs = next_obs

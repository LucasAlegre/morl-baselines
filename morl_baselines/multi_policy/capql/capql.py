"""CAPQL algorithm."""

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
import time
import wandb
from torch.distributions import Normal

from morl_baselines.common.buffer import TensorReplayBuffer
from morl_baselines.common.evaluation import (
    log_all_multi_policy_metrics,
    log_episode_info,
    policy_evaluation_mo,
)
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.networks import layer_init, mlp, polyak_update
from morl_baselines.common.weights import equally_spaced_weights


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class WeightSamplerAngle:
    """Sample weight vectors from normal distribution."""
 
    def __init__(self, rwd_dim, angle, w=None, device="cpu"):
        """Initialize the weight sampler."""
        self.rwd_dim = rwd_dim
        self.angle = angle
        self.device = device
        if w is None:
            w = th.ones(rwd_dim, device=device)
        else:
            w = th.as_tensor(w, device=device, dtype=th.float32)
        w = w / th.norm(w)
        self.w = w
 
    def sample(self, n_sample):
        """Sample n_sample weight vectors from normal distribution."""
        s = th.randn(n_sample, self.rwd_dim, device=self.device)
 
        # remove fluctuation on dir w
        s = s - (s @ self.w).view(-1, 1) * self.w.view(1, -1)
 
        # normalize it
        s = s / (th.norm(s, dim=1, keepdim=True) + 1e-8)
 
        # sample angle
        s_angle = th.rand(n_sample, 1, device=self.device) * self.angle
 
        # compute shifted vector from w
        w_sample = th.tan(s_angle) * s + self.w.view(1, -1)
 
        w_sample = w_sample / th.norm(w_sample, dim=1, keepdim=True, p=1)
 
        return w_sample


class Policy(nn.Module):
    """Policy network."""

    def __init__(self, obs_dim, rew_dim, output_dim, action_space, net_arch=[256, 256]):
        """Initialize the policy network."""
        super().__init__()
        self.action_space = action_space
        self.latent_pi = mlp(obs_dim + rew_dim, -1, net_arch)
        self.mean = nn.Linear(net_arch[-1], output_dim)
        self.log_std_linear = nn.Linear(net_arch[-1], output_dim)

        # action rescaling
        self.register_buffer("action_scale", th.tensor((action_space.high - action_space.low) / 2.0, dtype=th.float32))
        self.register_buffer("action_bias", th.tensor((action_space.high + action_space.low) / 2.0, dtype=th.float32))

        self.apply(layer_init)

    def forward(self, obs, w):
        """Forward pass of the policy network."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        h = self.latent_pi(th.concat((obs, w), dim=obs.dim() - 1))
        mean = self.mean(h)
        log_std = self.log_std_linear(h)
        log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def get_action(self, obs, w):
        """Get an action from the policy network."""
        mean, _ = self.forward(obs, w)
        return th.tanh(mean) * self.action_scale + self.action_bias

    def sample(self, obs, w):
        """Sample an action from the policy network using reparameterization trick."""
        mean, log_std = self.forward(obs, w)
        std = log_std.exp()
 
        # manual rsample
        noise = th.randn_like(mean)
        x_t = mean + std * noise
 
        # restrict the outputs
        y_t = th.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
 
        # manual log_prob calculation for Normal distribution
        # log_p = -0.5 * ( ((x-mean)/std)**2 + 2*log_std + log(2*pi) )
        log_prob = -0.5 * (noise.pow(2) + 2 * log_std + th.log(th.tensor(2 * np.pi, device=mean.device))).sum(dim=-1)
 
        # tanh correction: log_prob -= sum(log(scale * (1 - tanh(x)^2) + epsilon))
        log_prob -= th.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON).sum(dim=-1)
        log_prob = log_prob.clamp(-1e3, 1e3)
 
        mean_action = th.tanh(mean) * self.action_scale + self.action_bias
 
        return action, log_prob, mean_action


class QNetwork(nn.Module):
    """Q-network ensemble S x Ax W -> R^(num_q_nets * reward_dim)."""

    def __init__(self, obs_dim, action_dim, rew_dim, num_q_nets=2, net_arch=[256, 256]):
        """Initialize the Q-network ensemble."""
        super().__init__()
        self.num_q_nets = num_q_nets
        self.rew_dim = rew_dim
        self.net = mlp(obs_dim + action_dim + rew_dim, num_q_nets * rew_dim, net_arch)
        self.apply(layer_init)

    def forward(self, obs, action, w):
        """Forward pass of the Q-network ensemble."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if w.dim() == 1:
            w = w.unsqueeze(0)

        batch_size = obs.shape[0]
        q_values = self.net(th.cat((obs, action, w), dim=-1))
        # Reshape to (num_q_nets, batch_size, rew_dim) to match previous stack behavior
        return q_values.view(batch_size, self.num_q_nets, self.rew_dim).permute(1, 0, 2)


class CAPQL(MOAgent, MOPolicy):
    """CAPQL algorithm.

    MULTI-OBJECTIVE REINFORCEMENT LEARNING: CONVEXITY, STATIONARITY AND PARETO OPTIMALITY
    Haoye Lu, Daniel Herman & Yaoliang Yu
    ICLR 2023
    Paper: https://openreview.net/pdf?id=TjEzIsyEsQ6
    Code based on: https://github.com/haoyelu/CAPQL
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1000000,
        net_arch: List = [256, 256],
        batch_size: int = 128,
        num_q_nets: int = 2,
        alpha: float = 0.2,
        learning_starts: int = 1000,
        gradient_updates: int = 1,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "CAPQL",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
    ):
        """CAPQL algorithm with continuous actions.

        It extends the Soft-Actor Critic algorithm to multi-objective RL.
        It learns the policy and Q-networks conditioned on the weight vector.

        Args:
            env (gym.Env): The environment to train on.
            learning_rate (float, optional): The learning rate. Defaults to 3e-4.
            gamma (float, optional): The discount factor. Defaults to 0.99.
            tau (float, optional): The soft update coefficient. Defaults to 0.005.
            buffer_size (int, optional): The size of the replay buffer. Defaults to int(1e6).
            net_arch (List, optional): The network architecture for the policy and Q-networks. Defaults to [256, 256].
            batch_size (int, optional): The batch size for training. Defaults to 128.
            num_q_nets (int, optional): The number of Q-networks to use. Defaults to 2.
            alpha (float, optional): The entropy regularization coefficient. Defaults to 0.2.
            learning_starts (int, optional): The number of steps to take before starting to train. Defaults to 1000.
            gradient_updates (int, optional): The number of gradient steps to take per update. Defaults to 1.
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
        self.buffer_size = buffer_size
        self.num_q_nets = num_q_nets
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates
        self.alpha = alpha

        self.replay_buffer = TensorReplayBuffer(
            self.observation_shape,
            self.action_dim,
            rew_dim=self.reward_dim,
            max_size=self.buffer_size,
            device=self.device,
        )
 
        self.q_nets = QNetwork(
            self.observation_dim, self.action_dim, self.reward_dim, num_q_nets=num_q_nets, net_arch=net_arch
        ).to(self.device)
        self.target_q_nets = QNetwork(
            self.observation_dim, self.action_dim, self.reward_dim, num_q_nets=num_q_nets, net_arch=net_arch
        ).to(self.device)
 
        self.target_q_nets.load_state_dict(self.q_nets.state_dict())
        for param in self.target_q_nets.parameters():
            param.requires_grad = False
 
        self.policy = Policy(
            self.observation_dim, self.reward_dim, self.action_dim, self.action_space, net_arch=net_arch
        ).to(self.device)
 
        self.q_optim = optim.Adam(self.q_nets.parameters(), lr=self.learning_rate)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        self._n_updates = 0

        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def get_config(self):
        """Get the configuration of the agent."""
        return {
            "env_id": getattr(self.env.unwrapped.spec, "id", "Unknown"),
            "learning_rate": self.learning_rate,
            "num_q_nets": self.num_q_nets,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "gradient_updates": self.gradient_updates,
            "alpha": self.alpha,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
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
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = self.experiment_name if filename is None else filename
        th.save(saved_params, save_dir + "/" + filename + ".tar")

    def load(self, path, load_replay_buffer=True):
        """Load the agent weights from a file."""
        params = th.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(params["policy_state_dict"])
        self.policy_optim.load_state_dict(params["policy_optimizer_state_dict"])
        for i, (q_net, target_q_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            q_net.load_state_dict(params["q_net_" + str(i) + "_state_dict"])
            target_q_net.load_state_dict(params["target_q_net_" + str(i) + "_state_dict"])
        self.q_optim.load_state_dict(params["q_nets_optimizer_state_dict"])
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def _sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)

    def update(self):
        """Update the policy and the Q-nets."""
        for _ in range(self.gradient_updates):
            (s_obs, s_actions, w, s_rewards, s_next_obs, s_dones) = self._sample_batch_experiences()
 
            with th.no_grad():
                next_actions, log_pi, _ = self.policy.sample(s_next_obs, w)
                # Single forward pass for target ensemble
                q_targets = self.target_q_nets(s_next_obs, next_actions, w)
                min_target_q = th.min(q_targets, dim=0)[0] - self.alpha * log_pi.reshape(-1, 1)
                target_q = s_rewards + (1 - s_dones) * self.gamma * min_target_q
 
            # Single forward pass for critic ensemble
            q_values = self.q_nets(s_obs, s_actions, w)
            critic_loss = F.mse_loss(q_values, target_q.unsqueeze(0).expand_as(q_values))
 
            self.q_optim.zero_grad()
            critic_loss.backward()
            self.q_optim.step()
 
            # Policy update
            pi, log_pi, _ = self.policy.sample(s_obs, w)
            # Single forward pass for actor update
            q_values_pi = self.q_nets(s_obs, pi, w)
            min_q = th.min(q_values_pi, dim=0)[0]
 
            min_q = (min_q * w).sum(dim=-1, keepdim=True)
            policy_loss = ((self.alpha * log_pi.unsqueeze(-1)) - min_q).mean()
 
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()
 
            polyak_update(self.q_nets.parameters(), self.target_q_nets.parameters(), self.tau)
 
        if self.log and self.global_step % 1000 == 0:
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

        action = self.policy.get_action(obs, w)
 
        if not torch_action:
            action = action.detach().cpu().numpy()
        
        if action.ndim == 2 and action.shape[0] == 1:
            action = action.squeeze(0)
 
        return action

    def train(
        self,
        total_timesteps: int,
        eval_env: gymnasium.Env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        num_eval_weights_for_eval: int = 50,
        eval_freq: int = 10000,
        reset_num_timesteps: bool = False,
        checkpoints: bool = False,
        save_freq: int = 10000,
    ):
        """Train the agent."""
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_front": num_eval_weights_for_front,
                    "num_eval_episodes_for_front": num_eval_episodes_for_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "eval_freq": eval_freq,
                    "reset_num_timesteps": reset_num_timesteps,
                }
            )
 
        self.n_envs = self.env.num_envs if hasattr(self.env, "num_envs") else 1
        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)
 
        angle = th.pi * (22.5 / 180)
        weight_sampler = WeightSamplerAngle(self.reward_dim, angle, device=self.device)
 
        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
 
        start_time = time.time()
        obs, info = self.env.reset()
        for _ in range(1, (total_timesteps // self.n_envs) + 1):
            self.global_step += self.n_envs
 
            with th.no_grad():
                tensor_w = weight_sampler.sample(1).view(-1)
                w_batch = tensor_w.unsqueeze(0).expand(self.n_envs, -1)
 
                if self.global_step < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    obs_tensor = th.from_numpy(obs).to(self.device).float()
                    action = self.policy.get_action(obs_tensor, w_batch)
                    action = action.detach().cpu().numpy()
                    if self.n_envs == 1:
                        action = action.squeeze(0)
 
            next_obs, vector_reward, terminated, truncated, info = self.env.step(action)
 
            # add_batch handles n_envs > 1
            self.replay_buffer.add_batch(obs, action, vector_reward, next_obs, terminated, weights=w_batch)
 
            if self.global_step >= self.learning_starts:
                self.update()
 
            if self.n_envs == 1:
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    self.num_episodes += 1
                    if self.log and "episode" in info.keys():
                        log_episode_info(info["episode"], np.dot, w_np, self.global_step)
                else:
                    obs = next_obs
            else:
                # Vectorized envs handle reset automatically
                obs = next_obs
                if "final_info" in info:
                    for i, has_final in enumerate(info["_final_info"]):
                        if has_final:
                            self.num_episodes += 1
                            if self.log and "episode" in info["final_info"][i]:
                                w_np = tensor_w.detach().cpu().numpy()
                                log_episode_info(info["final_info"][i]["episode"], np.dot, w_np, self.global_step)
 
            if self.log and self.global_step % 1000 < self.n_envs:
                sps = int(self.global_step / (time.time() - start_time))
                if self.log:
                    wandb.log({"charts/SPS": sps}, commit=False)
                print(f"Step: {self.global_step}, SPS: {sps}")
 
            if self.log and self.global_step % eval_freq < self.n_envs:
                # Evaluation
                returns_test_tasks = [
                    policy_evaluation_mo(self, eval_env, ew, rep=num_eval_episodes_for_front)[3] for ew in eval_weights
                ]
                log_all_multi_policy_metrics(
                    current_front=returns_test_tasks,
                    hv_ref_point=ref_point,
                    reward_dim=self.reward_dim,
                    global_step=self.global_step,
                    n_sample_weights=num_eval_weights_for_eval,
                    ref_front=known_pareto_front,
                )
 
            # Checkpoint
            if checkpoints and self.global_step % save_freq < self.n_envs:
                self.save(filename=f"CAPQL step={self.global_step}", save_replay_buffer=False)
 
        self.close_wandb()

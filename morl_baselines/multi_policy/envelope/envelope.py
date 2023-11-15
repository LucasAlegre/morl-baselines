"""Envelope Q-Learning implementation.

Code modified to fit the MoDMSE environment
"""
import copy
import os
from typing import List, Optional, Union
from typing_extensions import override
from time import sleep
import json
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.networks import NatureCNN, mlp
from morl_baselines.common.prioritized_buffer import PrioritizedReplayBuffer
from morl_baselines.common.utils import (
    equally_spaced_weights,
    get_grad_norm,
    layer_init,
    linearly_decaying_value,
    log_all_multi_policy_metrics,
    log_episode_info,
    polyak_update,
    random_weights,
)
import wandb


class QNet(nn.Module):
    """Multi-objective Q-Network conditioned on the weight vector."""

    def __init__(self, obs_shape, action_dim, rew_dim, net_arch, custom_model=None):
        """Initialize the Q network.

        Args:
            obs_shape: shape of the observation
            action_dim: number of actions
            rew_dim: number of objectives
            net_arch: network architecture (number of units per layer)
            custom_model: custom model to use for the feature extractor (overrides the premade models, the input dimension must be obs_shape)
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.rew_dim = rew_dim
        self.custom_model = custom_model
        if custom_model is not None:
            # Dummy tensor to get the output dimension of the custom model
            dummy_obs = th.zeros((1,) + obs_shape)
            dummy_out = None
            try:
                dummy_out = custom_model(dummy_obs)
            except RuntimeError:
                print("The custom model must take the observation shape as input size")
            input_dim = dummy_out.view(dummy_out.size(0), -1).size(1) + rew_dim
            self.feature_extractor = custom_model
            self.model_is_custom = True
        else:
            if len(obs_shape) == 1:
                self.feature_extractor = None
                input_dim = obs_shape[0] + rew_dim
            elif len(obs_shape) > 1:  # Image observation
                self.feature_extractor = NatureCNN(self.obs_shape, features_dim=512)
                input_dim = self.feature_extractor.features_dim + rew_dim
        # |S| + |R| -> ... -> |A| * |R|
        self.net = mlp(input_dim, action_dim * rew_dim, net_arch)
        self.apply(layer_init)

    def forward(self, obs, w):
        """Predict Q values for all actions.

        Args:
            obs: current observation
            w: weight vector

        Returns: the Q values for all actions

        """
        if self.custom_model is not None:
            features = self.feature_extractor(obs)
            input = th.cat((features, w), dim=w.dim() - 1)
        else:
            if self.feature_extractor is not None:
                features = self.feature_extractor(obs / 255.0)
                input = th.cat((features, w), dim=w.dim() - 1)
            else:
                input = th.cat((obs, w), dim=w.dim() - 1)
        q_values = self.net(input)
        return q_values.view(-1, self.action_dim, self.rew_dim)  # Batch size X Actions X Rewards


class Envelope(MOPolicy, MOAgent):
    """Envelope Q-Leaning Algorithm.

    Envelope uses a conditioned network to embed multiple policies (taking the weight as input).
    The main change of this algorithm compare to a scalarized CN DQN is the target update.
    Paper: R. Yang, X. Sun, and K. Narasimhan, “A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation,” arXiv:1908.08342 [cs], Nov. 2019, Accessed: Sep. 06, 2021. [Online]. Available: http://arxiv.org/abs/1908.08342.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        initial_epsilon: float = 0.01,
        final_epsilon: float = 0.01,
        epsilon_decay_steps: int = None,  # None == fixed epsilon
        tau: float = 1.0,
        target_net_update_freq: int = 50,  # ignored if tau != 1.0
        buffer_size: int = int(1e2),
        net_arch: List = [256, 256],
        batch_size: int = 1, # 256
        learning_starts: int = 100,
        gradient_updates: int = 1,
        gamma: float = 0.99,
        max_grad_norm: Optional[float] = None,
        envelope: bool = True,
        num_sample_w: int = 1, # 4
        per: bool = True,
        per_alpha: float = 0.6,
        initial_homotopy_lambda: float = 0.0,
        final_homotopy_lambda: float = 1.0,
        homotopy_decay_steps: int = None,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "Envelope",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        custom_qnet: Optional[nn.Module] = None,
        action_masking: bool = False,
    ):
        """Envelope Q-learning algorithm.

        Args:
            env: The environment to learn from.
            learning_rate: The learning rate (alpha).
            initial_epsilon: The initial epsilon value for epsilon-greedy exploration.
            final_epsilon: The final epsilon value for epsilon-greedy exploration.
            epsilon_decay_steps: The number of steps to decay epsilon over.
            tau: The soft update coefficient (keep in [0, 1]).
            target_net_update_freq: The frequency with which the target network is updated.
            buffer_size: The size of the replay buffer.
            net_arch: The size of the hidden layers of the value net.
            batch_size: The size of the batch to sample from the replay buffer.
            learning_starts: The number of steps before learning starts i.e. the agent will be random until learning starts.
            gradient_updates: The number of gradient updates per step.
            gamma: The discount factor (gamma).
            max_grad_norm: The maximum norm for the gradient clipping. If None, no gradient clipping is applied.
            envelope: Whether to use the envelope method.
            num_sample_w: The number of weight vectors to sample for the envelope target.
            per: Whether to use prioritized experience replay.
            per_alpha: The alpha parameter for prioritized experience replay.
            initial_homotopy_lambda: The initial value of the homotopy parameter for homotopy optimization.
            final_homotopy_lambda: The final value of the homotopy parameter.
            homotopy_decay_steps: The number of steps to decay the homotopy parameter over.
            project_name: The name of the project, for wandb logging.
            experiment_name: The name of the experiment, for wandb logging.
            wandb_entity: The entity of the project, for wandb logging.
            log: Whether to log to wandb.
            seed: The seed for the random number generator.
            device: The device to use for training.
            custom_qnet: A custom Q network to use. (Must implement )
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
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
        self.action_masking = action_masking
        if custom_qnet is not None:
            self.q_net = custom_qnet.to(self.device)
            self.target_q_net = copy.deepcopy(custom_qnet).to(self.device)
        else:
            self.q_net = QNet(self.observation_shape, self.action_dim, self.reward_dim, net_arch=net_arch).to(self.device)
            self.target_q_net = QNet(self.observation_shape, self.action_dim, self.reward_dim, net_arch=net_arch).to(self.device)
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        for param in self.target_q_net.parameters():
            param.requires_grad = False

        self.q_optim = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

        self.envelope = envelope
        self.num_sample_w = num_sample_w
        self.homotopy_lambda = self.initial_homotopy_lambda
        self.reward_dim = 5

        self.rand = True

        obs = self.env.obs_to_numpy(self.env.get_observation())

        self.observation_shape_schedule = obs[0].shape
        self.observation_shape_ticket = obs[1].shape

        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.observation_shape_schedule,
                self.observation_shape_ticket,
                action_dim=1,
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
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    @override
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
            "seed": self.seed,
            "action_masking": self.action_masking,
        }

    def save(self, save_replay_buffer: bool = True, save_dir: str = "weights/", filename: Optional[str] = None):
        """Save the model and the replay buffer if specified.

        Args:
            save_replay_buffer: Whether to save the replay buffer too.
            save_dir: Directory to save the model.
            filename: filename to save the model.
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        self.q_net.save_weights(save_dir + "q_net_")
        self.target_q_net.save_weights(save_dir + "target_q_net_")

        saved_params = {}
        saved_params["q_net_optimizer_state_dict"] = self.q_optim.state_dict()
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = "envelope_params"
        th.save(saved_params, save_dir + "/" + filename + ".tar")

    def load(self, path: str = "weights/", load_replay_buffer: bool = True):
        """Load the model and the replay buffer if specified.

        Args:
            path: Path to the model.
            load_replay_buffer: Whether to load the replay buffer too.
        """
        params = th.load(path+"envelope_params.tar")
        self.q_net.load_weights(path + "q_net_model_weights.pth")
        self.target_q_net.load_weights(path + "target_q_net_model_weights.pth")
        self.q_optim.load_state_dict(params["q_net_optimizer_state_dict"])
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def unpack(self, obs):
        # We split the obs into the schedule and the ticket list
        # obs_schedule is the whole obs except the last 15 elements
        obs_schedule = obs[:-15]
        # obs_ticket_list is the last 15 elements of obs
        obs_ticket_list = obs[-15:]
        # We reshape the schedule
        obs_schedule = obs_schedule.reshape((5, 336, 3))
        # We put the obs back into a dict
        obs = {"schedule": obs_schedule, "ticket_list": obs_ticket_list}
        return obs
    
    def unpack_tolist(self, obs_vec):
        final = []
        for obs in obs_vec:
            final.append(self.unpack(obs))
        return final

    def repeat_list(self, list, n):
        repeated_list = [item for item in list for _ in range(n)]
        return repeated_list

    def obs_to_tensor(self, obs):
        obs_schedule = obs["schedule"]
        obs_ticket_list = obs["ticket_list"]

        obs_schedule = th.tensor(obs_schedule).float().to(self.device)
        obs_ticket_list = th.tensor(obs_ticket_list).float().to(self.device)

        obs = {"schedule": obs_schedule, "ticket_list": obs_ticket_list}
        return obs

    def obs_list_to_tensor(self, obs_list):
        final = []
        for obs in obs_list:
            final.append(self.obs_to_tensor(obs))
        return final

    def __sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)

    @override
    def update(self):
        critic_losses = []
        for g in range(self.gradient_updates):
            if self.per:
                (
                    b_obs_schedule,
                    b_obs_ticket,
                    b_actions,
                    b_rewards,
                    b_next_obs_schedule,
                    b_next_obs_ticket,
                    b_dones,
                    b_inds,
                ) = self.__sample_batch_experiences()
            else:
                (
                    b_obs,
                    b_actions,
                    b_rewards,
                    b_next_obs,
                    b_dones,
                ) = self.__sample_batch_experiences()

            sampled_w = (
                th.tensor(random_weights(dim=self.reward_dim, n=self.num_sample_w, dist="dirichlet", rng=self.np_random))
                .float()
                .to(self.device)
            )  # sample num_sample_w random weights
            w = sampled_w.repeat_interleave(b_obs_schedule.size(0), 0)  # repeat the weights for each sample
            b_obs_schedule, b_obs_ticket, b_actions, b_rewards, b_next_obs_schedule, b_next_obs_ticket, b_dones = (
                b_obs_schedule.repeat(self.num_sample_w, 1, 1, 1),
                b_obs_ticket.repeat(self.num_sample_w, 1),
                b_actions.repeat(self.num_sample_w, 1),
                b_rewards.repeat(self.num_sample_w, 1),
                b_next_obs_schedule.repeat(self.num_sample_w, 1, 1, 1),
                b_next_obs_ticket.repeat(self.num_sample_w, 1),
                b_dones.repeat(self.num_sample_w, 1),
            )
            with th.no_grad():
                if self.envelope:
                    target = self.envelope_target(b_next_obs_schedule, b_next_obs_ticket, w, sampled_w)
                else:
                    target = self.ddqn_target(b_next_obs, w)
                target_q = b_rewards + (1 - b_dones) * self.gamma * target

            q_values = self.q_net(b_obs_schedule, b_obs_ticket, w)
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
                wandb.log(
                    {
                        "losses/grad_norm": get_grad_norm(self.q_net.parameters()).item(),
                        "global_step": self.global_step,
                    },
                )
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.q_optim.step()
            critic_losses.append(critic_loss.item())

            if self.per:
                td_err = (q_value[: len(b_inds)] - target_q[: len(b_inds)]).detach()
                priority = th.einsum("sr,sr->s", td_err, w[: len(b_inds)]).abs()
                priority = priority.cpu().numpy().flatten()
                priority = (priority + self.replay_buffer.min_priority) ** self.per_alpha
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
            wandb.log(
                {
                    "losses/critic_loss": np.mean(critic_losses),
                    "metrics/epsilon": self.epsilon,
                    "metrics/homotopy_lambda": self.homotopy_lambda,
                    "global_step": self.global_step,
                },
            )
            if self.per:
                wandb.log({"metrics/mean_priority": np.mean(priority)})

    @override
    def eval(self, obs_schedule: np.ndarray, obs_ticket: np.ndarray, w: np.ndarray) -> int:
        obs_schedule = th.as_tensor(obs_schedule).float().to(self.device)
        obs_ticket = th.as_tensor(obs_ticket).float().to(self.device)
        w = th.as_tensor(w).float().to(self.device)
        return self.max_action(obs_schedule, obs_ticket, w)

    def act(self, obs_schedule: th.Tensor, obs_ticket: th.Tensor, w: th.Tensor) -> int:
        """Epsilon-greedily select an action given an observation and weight.

        Args:
            obs: observation
            w: weight vector

        Returns: an integer representing the action to take.
        """
        if self.np_random.random() < self.epsilon:
            self.rand = True
            return self.env.action_space.sample_legal_action(self.env.state)
        else:
            self.rand = False
            return self.max_action(obs_schedule, obs_ticket, w)

    @th.no_grad()
    def max_action(self, obs_schedule: th.Tensor, obs_ticket: th.Tensor, w: th.Tensor) -> int:
        """Select the action with the highest Q-value given an observation and weight.

        Args:
            obs: observation
            w: weight vector

        Returns: the action with the highest Q-value.
        """

        q_values = self.q_net(obs_schedule, obs_ticket, w)
        scalarized_q_values = th.einsum("r,bar->ba", w, q_values)

        if self.action_masking:
            legal_filter = self.env.action_space.legal_filter(self.env.state)
            old_shape = scalarized_q_values.shape
            scalarized_q_values = scalarized_q_values.flatten()
            scalarized_q_values[legal_filter == 0] = -(2**62)
            scalarized_q_values = scalarized_q_values.reshape(old_shape)
        max_act = th.argmax(scalarized_q_values, dim=1)
        return max_act.detach().item()

    @th.no_grad()
    def envelope_target(self, obs_schedule: th.Tensor, obs_ticket: th.Tensor, w: th.Tensor, sampled_w: th.Tensor) -> th.Tensor:
        """Computes the envelope target for the given observation and weight.

        Args:
            obs: current observation.
            w: current weight vector.
            sampled_w: set of sampled weight vectors (>1!).

        Returns: the envelope target.
        """
        # Repeat the weights for each sample
        W = sampled_w.unsqueeze(0).repeat(obs_schedule.size(0), 1, 1)
        # Repeat the observations for each sampled weight
        next_obs_schedule = obs_schedule.unsqueeze(1).repeat(1, sampled_w.size(0), 1, 1, 1)
        next_obs_ticket = obs_ticket.unsqueeze(1).repeat(1, sampled_w.size(0), 1)

        # Batch size X Num sampled weights X Num actions X Num objectives
        next_q_values = self.q_net(next_obs_schedule, next_obs_ticket, W).view(obs_schedule.size(0), sampled_w.size(0), self.action_dim, self.reward_dim)
        # Scalarized Q values for each sampled weight
        scalarized_next_q_values = th.einsum("br,bwar->bwa", w, next_q_values)
        # Max Q values for each sampled weight
        max_q, ac = th.max(scalarized_next_q_values, dim=2)
        # Max weights in the envelope
        pref = th.argmax(max_q, dim=1)

        # MO Q-values evaluated on the target network
        next_q_values_target = self.target_q_net(next_obs_schedule, next_obs_ticket, W).view(
            obs_schedule.size(0), sampled_w.size(0), self.action_dim, self.reward_dim
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
        """Double DQN target for the given observation and weight.

        Args:
            obs: observation
            w: weight vector.

        Returns: the DQN target.
        """
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
        eval_env: Optional[gym.Env] = None,
        ref_point: Optional[np.ndarray] = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        weight: Optional[np.ndarray] = None,
        total_episodes: Optional[int] = None,
        reset_num_timesteps: bool = True,
        eval_freq: int = 10000,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        reset_learning_starts: bool = False,
    ):
        """Train the agent.

        Args:
            total_timesteps: total number of timesteps to train for.
            eval_env: environment to use for evaluation. If None, it is ignored.
            ref_point: reference point for the hypervolume computation.
            known_pareto_front: known pareto front for the hypervolume computation.
            weight: weight vector. If None, it is randomly sampled every episode (as done in the paper).
            total_episodes: total number of episodes to train for. If None, it is ignored.
            reset_num_timesteps: whether to reset the number of timesteps. Useful when training multiple times.
            eval_freq: policy evaluation frequency (in number of steps).
            num_eval_weights_for_front: number of weights to sample for creating the pareto front when evaluating.
            num_eval_episodes_for_front: number of episodes to run when evaluating the policy.
            reset_learning_starts: whether to reset the learning starts. Useful when training multiple times.
        """
        if eval_env is not None:
            assert ref_point is not None, "Reference point must be provided for the hypervolume computation."
        if self.log:
            self.register_additional_config({"ref_point": ref_point.tolist(), "known_front": known_pareto_front})

        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        if reset_learning_starts:  # Resets epsilon-greedy exploration
            self.learning_starts = self.global_step

        num_episodes = 0
        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)
        obs, _ = self.env.reset()

        w = weight if weight is not None else random_weights(self.reward_dim, 1, dist="dirichlet", rng=self.np_random)
        tensor_w = th.tensor(w).float().to(self.device)

        for _ in range(1, total_timesteps + 1):
            if total_episodes is not None and num_episodes == total_episodes:
                break

            if self.global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.act(th.as_tensor(obs).float().to(self.device), tensor_w)

            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)
            self.global_step += 1

            self.replay_buffer.add(obs, action, vec_reward, next_obs, terminated)
            if self.global_step >= self.learning_starts:
                self.update()

            if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                current_front = [
                    self.policy_eval(eval_env, weights=ew, num_episodes=num_eval_episodes_for_front, writer=None)[3]
                    for ew in eval_weights
                ]
                log_all_multi_policy_metrics(
                    current_front=current_front,
                    hv_ref_point=ref_point,
                    reward_dim=self.reward_dim,
                    global_step=self.global_step,
                    writer=self.writer,
                    ref_front=known_pareto_front,
                )

            if terminated or truncated:
                obs, _ = self.env.reset()
                num_episodes += 1
                self.num_episodes += 1

                if self.log and "episode" in info.keys():
                    log_episode_info(info["episode"], np.dot, w, self.global_step, writer=self.writer)

                if weight is None:
                    w = random_weights(self.reward_dim, 1, dist="dirichlet", rng=self.np_random)
                    tensor_w = th.tensor(w).float().to(self.device)

            else:
                obs = next_obs
    


    def train_scheduling(
        self,
        eval_env: Optional[gym.Env] = None,
        ref_point: Optional[np.ndarray] = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        weight: Optional[np.ndarray] = None,
        total_schedulings: Optional[int] = None,
        reset_num_timesteps: bool = True,
        eval_freq: int = 10000,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        reset_learning_starts: bool = False,
        exp_dir: str = "experiments",
    ):
        """Train the agent for scheduling.

        Args:
            eval_env: environment to use for evaluation. If None, it is ignored.
            ref_point: reference point for the hypervolume computation.
            known_pareto_front: known pareto front for the hypervolume computation.
            weight: weight vector. If None, it is randomly sampled every episode (as done in the paper).
            total_schedulings: total number of episodes to train for. If None, it is ignored.
            reset_num_timesteps: whether to reset the number of timesteps. Useful when training multiple times.
            eval_freq: policy evaluation frequency (in number of steps).
            num_eval_weights_for_front: number of weights to sample for creating the pareto front when evaluating.
            num_eval_episodes_for_front: number of episodes to run when evaluating the policy.
            reset_learning_starts: whether to reset the learning starts. Useful when training multiple times.
        """

        if eval_env is not None:
            assert ref_point is not None, "Reference point must be provided for the hypervolume computation."
        # if self.log:
        #     self.register_additional_config({"ref_point": ref_point.tolist(), "known_front": known_pareto_front})

        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        if reset_learning_starts:  # Resets epsilon-greedy exploration
            self.learning_starts = self.global_step

        num_episodes = 0
        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)
        obs, _ = self.env.reset()
        obs_schedule, obs_ticket = obs
        w = weight if weight is not None else random_weights(self.reward_dim, 1, dist="dirichlet", rng=self.np_random)
        tensor_w = th.tensor(w).float().to(self.device)


        # Load json conf from "experiments/exp_conf.json"
        with open('experiments/exp_conf.json') as json_file:
            exp_conf = json.load(json_file)
        
        # Get the experiment name
        exp_name = exp_conf["exp_name"]+"_"+str(exp_conf["exp_id"])

        # Rewrite the json file with the new experiment id incermented
        exp_conf["exp_id"] = exp_conf["exp_id"]+1
        with open('experiments/exp_conf.json', 'w') as outfile:
            json.dump(exp_conf, outfile)

        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        
        if not os.path.isdir(exp_dir+"/"+exp_name):
            os.makedirs(exp_dir+"/"+exp_name)


        env_state = self.env.state


        terminated, truncated = False, False
        # Same thing, different names for comprehension
        total_steps = env_state['global']['n_steps']

        total_rewards = []

        # if not os.path.isdir(exp_dir+"/"+exp_name+"/rewards"):
        #     os.makedirs(exp_dir+"/"+exp_name+"/rewards")

        scheduling = 0
        
        while scheduling < total_schedulings:
            print("Scheduling "+ str(scheduling+1)+"/"+str(total_schedulings))
            episode_rewards = []
            rts = 0

            #print("Real timesteps: "+str(rts+1)+"/"+str(total_steps)+" steps")

            # if not os.path.isdir(exp_dir+"/"+exp_name+"/rewards/schedule_"+str(scheduling)):
            #     os.makedirs(exp_dir+"/"+exp_name+"/rewards/schedule_"+str(scheduling))
            vec_reward = np.zeros(5)
            while not(self.env.episode_done()) and not(terminated or truncated): # An episode is completing a whole schedule over multiple timesteps
                steps_rewards = []
                self.env.get_new_tickets()
                self.env.loop_reset()
                ts = 0
                while not(self.env.loop_done()) and not(terminated or truncated): # A loop is creating a whole schedule at a timestep
                    
                    current_schedule = self.env.state['loop']['current_schedule']
                    current_step = self.env.state['loop']['current_step']
                    current_worker = self.env.state['loop']['current_worker']

                    # if there's already a ticket in the current step and current worker, we don't do anything
                    if current_schedule[current_step][current_worker].idx != -1:
                        # put here the steps updates for the env
                        self.env.state["loop"]["current_step"] += 1
                        self.env.state["episode"]["current_global_step"] += 1
                        ts += 1
                        self.env.loop_pass_done()
                        continue

                    # if there was a task finishing on the previous step, we put a slack to let the worker return to station
                    if current_schedule[current_step-1][current_worker].idx != -1:
                        action = 0

                    # if we don't learn yet we just sample random actions
                    if self.global_step < self.learning_starts:
                        action = self.env.action_space.sample_legal_action(self.env.state)
                    else:

                        # If we are in the past we do previous actions
                        if self.env.state['episode']['current_timestep'] > self.env.state['loop']['current_step']:
                            action = 1
                        
                        # If there isn't any ticket left we put a slack
                        if self.env.state['loop']['remaining_tickets_list'] == []:
                            self.env.state["loop"]["current_step"] = self.env.n_steps
                            self.env.state["loop"]["current_worker"] = self.env.n_technicians -1
                            self.env.loop_pass_done()
                            continue
                        # Else we act with the agent
                        else:
                            action = self.act(th.as_tensor(obs_schedule).float().to(self.device), th.as_tensor(obs_ticket).float().to(self.device), tensor_w)

                    next_obs, vec_reward, terminated, truncated, info = self.env.step(action)
                    next_obs_schedule, next_obs_ticket = next_obs
                    self.global_step += 1

                    self.replay_buffer.add(obs_schedule, obs_ticket, action, vec_reward, next_obs_schedule, next_obs_ticket, terminated)

                    if self.global_step >= self.learning_starts:
                        self.update()

                    # Evaluate the policy if needed
                    
                    # if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                    #     current_front = [
                    #         self.policy_eval(eval_env, weights=ew, num_episodes=num_eval_episodes_for_front, writer=None)[3]
                    #         for ew in eval_weights
                    #     ]
                    #     log_all_multi_policy_metrics(
                    #         current_front=current_front,
                    #         hv_ref_point=ref_point,
                    #         reward_dim=self.reward_dim,
                    #         global_step=self.global_step,
                    #         writer=self.writer,
                    #         ref_front=known_pareto_front,
                    #     )

                    if terminated or truncated:
                        print("Scheduling done")
                        obs, _ = self.env.reset()
                        obs_schedule, obs_ticket = obs
                        num_episodes += 1
                        self.num_episodes += 1
                        if weight is None:
                            w = random_weights(self.reward_dim, 1, dist="dirichlet", rng=self.np_random)
                            tensor_w = th.tensor(w).float().to(self.device)
                    else:
                        obs = next_obs
                        obs_schedule, obs_ticket = obs


                    

                    # If the file does not exist, create it and write the first line

                    # if not os.path.isfile(exp_dir+"/"+exp_name+"/rewards/schedule_"+str(scheduling)+"/rts_"+str(rts)+".csv"):
                    #     with open(exp_dir+"/"+exp_name+"/rewards/schedule_"+str(scheduling)+"/rts_"+str(rts)+".csv", 'w') as f:
                    #         f.write(','.join(map(str, np.array(vec_reward).flatten())))
                    # else:
                    # with open(exp_dir+"/"+exp_name+"/rewards/schedule_"+str(scheduling)+"/rts_"+str(rts)+".csv", 'a') as f:
                    #     f.write(','.join(map(str, np.array(vec_reward).flatten()))+"\n")

                    if self.log:
                        wandb.log(
                            {
                                "progress/scheduling":scheduling,
                                "progress/rts":self.env.state['episode']['current_timestep'],
                                "progress/step": self.env.state['loop']['current_step'],
                                "progress/worker": self.env.state['loop']['current_worker'],
                                "action/action": action,
                                "tickets/total": len(self.env.state['episode']['ticket_list']),
                                "tickets/remaining": len(self.env.state['loop']['remaining_tickets_list']),
                                "reward/makespan":vec_reward[0],
                                "reward/priority":vec_reward[1],
                                "reward/stability":vec_reward[2],
                                "reward/robustness":vec_reward[3],
                                "reward/timetotreatment":vec_reward[4],
                                "reward/scalarized":np.dot(vec_reward, w),
                                "global_step": self.global_step,
                            },
                        )
                    ts += 1

                    self.env.loop_pass_done()

                rts +=1
                self.save(save_dir=exp_dir+"/"+exp_name+"/weights/")

                self.env.render_to_csv(save_dir=exp_dir+"/"+exp_name+"/")
                if not Path(exp_dir+"/"+exp_name+"/rewards.csv").is_file():
                    with open(exp_dir+"/"+exp_name+"/rewards.csv", 'w') as f:
                        f.write("episode,makespan,priority,stability,robustness,timetotreatment\n")

                with open(exp_dir+"/"+exp_name+"/rewards.csv", 'a') as f:
                    f.write(str(self.env.state['episode']['current_timestep']-1)+","+(','.join(map(str, np.array(vec_reward).flatten()))+"\n"))

            obs, _ = self.env.reset()
            obs_schedule, obs_ticket = obs
            num_episodes += 1
            self.num_episodes += 1
            if weight is None:
                w = random_weights(self.reward_dim, 1, dist="dirichlet", rng=self.np_random)
                tensor_w = th.tensor(w).float().to(self.device)

            terminated, truncated = False, False
            scheduling += 1 

    
    def eval_scheduling_env(
        self,
        weight: np.ndarray = np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        total_schedulings: int = 1,
        exp_dir: str = "eval_experiments"
    ):
        
        '''Args:
            weight: weight vector. If None, it is randomly sampled every episode (as done in the paper).
            total_schedulings: total number of episodes to train for. If None, it is ignored.
        '''

        # if eval_env is not None:
        #     assert ref_point is not None, "Reference point must be provided for the hypervolume computation."
        # # if self.log:
        # #     self.register_additional_config({"ref_point": ref_point.tolist(), "known_front": known_pareto_front})


        num_episodes = 0
        # eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)
        obs, _ = self.env.reset()
        obs_schedule, obs_ticket = obs
        if weight is None:
            weight = random_weights(self.reward_dim, 1, dist="dirichlet", rng=self.np_random)
        w = weight
        tensor_w = th.tensor(w).float().to(self.device)

        if self.log:
            self.register_additional_config({"weight":
                                         {
                                                "makespan": w[0],
                                                "priority": w[1],
                                                "stability": w[2],
                                                "robustness": w[3],
                                                "timetotreatment": w[4]
                                         }})
        

        # Load json conf from "eval_experiments/exp_conf.json"
        with open('eval_experiments/exp_conf.json') as json_file:
            exp_conf = json.load(json_file)
        
        # Get the experiment name
        exp_name = exp_conf["exp_name"]+"_"+str(exp_conf["exp_id"])

        # Rewrite the json file with the new experiment id incermented
        exp_conf["exp_id"] = exp_conf["exp_id"]+1
        with open('eval_experiments/exp_conf.json', 'w') as outfile:
            json.dump(exp_conf, outfile)

        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        
        if not os.path.isdir(exp_dir+"/"+exp_name):
            os.makedirs(exp_dir+"/"+exp_name)


        env_state = self.env.state


        terminated, truncated = False, False
        # Same thing, different names for comprehension
        total_steps = env_state['global']['n_steps']

        total_rewards = []

        scheduling = 0
        
        while scheduling < total_schedulings:
            # print("Scheduling "+ str(scheduling+1)+"/"+str(total_schedulings))
            episode_rewards = []
            rts = 0

            vec_reward = np.zeros(5)
            while not(self.env.episode_done()) and not(terminated or truncated): # An episode is completing a whole schedule over multiple timesteps
                steps_rewards = []
                self.env.get_new_tickets()
                self.env.loop_reset()
                ts = 0
                while not(self.env.loop_done()) and not(terminated or truncated): # A loop is creating a whole schedule at a timestep
                    
                    current_schedule = self.env.state['loop']['current_schedule']
                    current_step = self.env.state['loop']['current_step']
                    current_worker = self.env.state['loop']['current_worker']

                    # if there's already a ticket in the current step and current worker, we don't do anything
                    if current_schedule[current_step][current_worker].idx != -1:
                        # put here the steps updates for the env
                        self.env.state["loop"]["current_step"] += 1
                        self.env.state["episode"]["current_global_step"] += 1
                        ts += 1
                        self.env.loop_pass_done()
                        continue

                    # if there was a task finishing on the previous step, we put a slack to let the worker return to station
                    if current_schedule[current_step-1][current_worker].idx != -1:
                        action = 0

                    # If we are in the past we do previous actions
                    if self.env.state['episode']['current_timestep'] > self.env.state['loop']['current_step']:
                        action = 1
                    
                    # If there isn't any ticket left we put a slack
                    if self.env.state['loop']['remaining_tickets_list'] == []:
                        self.env.state["loop"]["current_step"] = self.env.n_steps
                        self.env.state["loop"]["current_worker"] = self.env.n_technicians -1
                        self.env.loop_pass_done()
                        continue
                    # Else we act with the agent
                    else:
                        action = self.act(th.as_tensor(obs_schedule).float().to(self.device), th.as_tensor(obs_ticket).float().to(self.device), tensor_w)

                    next_obs, vec_reward, terminated, truncated, info = self.env.step(action)
                    next_obs_schedule, next_obs_ticket = next_obs
                    self.global_step += 1

                    # Evaluate the policy if needed
                    
                    # if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                    #     current_front = [
                    #         self.policy_eval(eval_env, weights=ew, num_episodes=num_eval_episodes_for_front, writer=None)[3]
                    #         for ew in eval_weights
                    #     ]
                    #     log_all_multi_policy_metrics(
                    #         current_front=current_front,
                    #         hv_ref_point=ref_point,
                    #         reward_dim=self.reward_dim,
                    #         global_step=self.global_step,
                    #         writer=self.writer,
                    #         ref_front=known_pareto_front,
                    #     )
                    if terminated or truncated:
                        print("Scheduling done")
                        obs, _ = self.env.reset()
                        obs_schedule, obs_ticket = obs
                        num_episodes += 1
                        self.num_episodes += 1
                    else:
                        obs = next_obs
                        obs_schedule, obs_ticket = obs

                    if self.log:
                        wandb.log(
                            {
                                "progress/scheduling":scheduling,
                                "progress/rts":self.env.state['episode']['current_timestep'],
                                "progress/step": self.env.state['loop']['current_step'],
                                "progress/worker": self.env.state['loop']['current_worker'],
                                "action/action": action,
                                "tickets/total": len(self.env.state['episode']['ticket_list']),
                                "tickets/remaining": len(self.env.state['loop']['remaining_tickets_list']),
                                "reward/makespan":vec_reward[0],
                                "reward/priority":vec_reward[1],
                                "reward/stability":vec_reward[2],
                                "reward/robustness":vec_reward[3],
                                "reward/timetotreatment":vec_reward[4],
                                "reward/scalarized":np.dot(vec_reward, w),
                                "global_step": self.global_step,
                            },
                        )
                    ts += 1
                    self.env.loop_pass_done()
                rts +=1
                self.save(save_dir=exp_dir+"/"+exp_name+"/weights/")
                self.env.render_to_csv(save_dir=exp_dir+"/"+exp_name+"/")
                if not Path(exp_dir+"/"+exp_name+"/rewards.csv").is_file():
                    with open(exp_dir+"/"+exp_name+"/rewards.csv", 'w') as f:
                        f.write("episode,makespan,priority,stability,robustness,timetotreatment\n")
                with open(exp_dir+"/"+exp_name+"/rewards.csv", 'a') as f:
                    f.write(str(self.env.state['episode']['current_timestep']-1)+","+(','.join(map(str, np.array(vec_reward).flatten()))+"\n"))

            obs, _ = self.env.reset()
            obs_schedule, obs_ticket = obs
            num_episodes += 1
            self.num_episodes += 1
            terminated, truncated = False, False
            scheduling += 1 
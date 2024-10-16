"""Multi-Objective PPO Algorithm."""

import time
from copy import deepcopy
from typing import List, Optional, Union
from typing_extensions import override

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import torch as th
import wandb
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
from torch import nn, optim
from torch.distributions import Normal

from morl_baselines.common.evaluation import log_episode_info
from morl_baselines.common.morl_algorithm import MOPolicy
from morl_baselines.common.networks import layer_init, mlp


class PPOReplayBuffer:
    """Replay buffer."""

    def __init__(
        self,
        size: int,
        num_envs: int,
        obs_shape: tuple,
        action_shape: tuple,
        reward_dim: int,
        device: Union[th.device, str],
    ):
        """Initialize the replay buffer.

        Args:
            size: Buffer size
            num_envs: Number of environments (for VecEnv)
            obs_shape: Observation shape
            action_shape: Action shape
            reward_dim: Reward dimension
            device: Device where the tensors are stored
        """
        self.size = size
        self.ptr = 0
        self.num_envs = num_envs
        self.device = device
        self.obs = th.zeros((self.size, self.num_envs) + obs_shape).to(device)
        self.actions = th.zeros((self.size, self.num_envs) + action_shape).to(device)
        self.logprobs = th.zeros((self.size, self.num_envs)).to(device)
        self.rewards = th.zeros((self.size, self.num_envs, reward_dim), dtype=th.float32).to(device)
        self.dones = th.zeros((self.size, self.num_envs)).to(device)
        self.values = th.zeros((self.size, self.num_envs, reward_dim), dtype=th.float32).to(device)

    def add(self, obs, actions, logprobs, rewards, dones, values):
        """Add a bunch of new transition to the buffer. (VecEnv makes more transitions at once).

        Args:
            obs: Observations
            actions: Actions
            logprobs: Log probabilities of the actions
            rewards: Rewards
            dones: Done signals
            values: Values
        """
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.logprobs[self.ptr] = logprobs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.ptr = (self.ptr + 1) % self.size

    def get(self, step: int):
        """Get data from the buffer at a specific step.

        Args:
            step: step

        Returns: A tuple of (obs, actions, logprobs, rewards, dones, values)

        """
        return (
            self.obs[step],
            self.actions[step],
            self.logprobs[step],
            self.rewards[step],
            self.dones[step],
            self.values[step],
        )

    def get_all(self):
        """Get all data from the buffer.

        Returns: A tuple of (obs, actions, logprobs, rewards, dones, values) containing all the data in the buffer.
        """
        return (
            self.obs,
            self.actions,
            self.logprobs,
            self.rewards,
            self.dones,
            self.values,
        )


def make_env(env_id, seed, idx, run_name, gamma):
    """Returns a function to create environments. This is because PPO works better with vectorized environments. Also, some tricks like clipping and normalizing the environments' features are applied.

    Args:
        env_id: Environment ID (for MO-Gymnasium)
        seed: Seed
        idx: Index of the environment
        run_name: Name of the run
        gamma: Discount factor

    Returns:
        A function to create environments
    """

    def thunk():
        if idx == 0:
            env = mo_gym.make(env_id, render_mode="rgb_array")
        else:
            env = mo_gym.make(env_id)
        reward_dim = env.unwrapped.reward_space.shape[0]
        """ if idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}_{seed}",
                episode_trigger=lambda e: e % 1000 == 0,
            ) """
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        for o in range(reward_dim):
            env = mo_gym.wrappers.MONormalizeReward(env, idx=o, gamma=gamma)
            env = mo_gym.wrappers.MOClipReward(env, idx=o, min_r=-10, max_r=10)
        env = MORecordEpisodeStatistics(env, gamma=gamma)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def _hidden_layer_init(layer):
    layer_init(layer, weight_gain=np.sqrt(2), bias_const=0.0)


def _critic_init(layer):
    layer_init(layer, weight_gain=1.0)


def _value_init(layer):
    layer_init(layer, weight_gain=0.01)


class MOPPONet(nn.Module):
    """Actor-Critic network."""

    def __init__(
        self,
        obs_shape: tuple,
        action_shape: tuple,
        reward_dim: int,
        net_arch: List = [64, 64],
    ):
        """Initialize the network.

        Args:
            obs_shape: Observation shape
            action_shape: Action shape
            reward_dim: Reward dimension
            net_arch: Number of units per layer
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.reward_dim = reward_dim
        self.net_arch = net_arch

        # S -> ... -> |R| (multi-objective)
        self.critic = mlp(
            input_dim=np.array(self.obs_shape).prod(),
            output_dim=self.reward_dim,
            net_arch=net_arch,
            activation_fn=nn.Tanh,
        )
        self.critic.apply(_hidden_layer_init)
        _critic_init(list(self.critic.modules())[-1])

        # S -> ... -> A (continuous)
        self.actor_mean = mlp(
            input_dim=np.array(self.obs_shape).prod(),
            output_dim=np.array(self.action_shape).prod(),
            net_arch=net_arch,
            activation_fn=nn.Tanh,
        )
        self.actor_mean.apply(_hidden_layer_init)
        _value_init(list(self.actor_mean.modules())[-1])
        self.actor_logstd = nn.Parameter(th.zeros(1, np.array(self.action_shape).prod()))

    def get_value(self, obs):
        """Get the value of an observation.

        Args:
            obs: Observation

        Returns: The predicted value of the observation.
        """
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        """Get the action and value of an observation.

        Args:
            obs: Observation
            action: Action. If None, a new action is sampled.

        Returns: A tuple of (action, logprob, entropy, value)
        """
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = th.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(obs),
        )


class MOPPO(MOPolicy):
    """Modified PPO to have a multi-objective value net (returning a vector) and applying weighted sum scalarization.

    This code has been adapted from the PPO implementation of clean RL https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
    """

    def __init__(
        self,
        id: int,
        networks: MOPPONet,
        weights: np.ndarray,
        envs: gym.vector.SyncVectorEnv,
        log: bool = False,
        steps_per_iteration: int = 2048,
        num_minibatches: int = 32,
        update_epochs: int = 10,
        learning_rate: float = 3e-4,
        gamma: float = 0.995,
        anneal_lr: bool = False,
        clip_coef: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        clip_vloss: bool = True,
        max_grad_norm: float = 0.5,
        norm_adv: bool = True,
        target_kl: Optional[float] = None,
        gae: bool = True,
        gae_lambda: float = 0.95,
        device: Union[th.device, str] = "auto",
        seed: int = 42,
        rng: Optional[np.random.Generator] = None,
    ):
        """Multi-objective PPO.

        Args:
            id: Policy ID
            networks: Actor-Critic networks
            weights: Weights of the objectives
            envs: Vectorized environments
            log: Whether to log
            steps_per_iteration: Number of steps per iteration
            num_minibatches: Number of minibatches
            update_epochs: Number of epochs to update the network
            learning_rate: Learning rate
            gamma: Discount factor
            anneal_lr: Whether to anneal the learning rate
            clip_coef: PPO clipping coefficient
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            clip_vloss: Whether to clip the value loss
            max_grad_norm: Maximum gradient norm
            norm_adv: Whether to normalize the advantage
            target_kl: Target KL divergence
            gae: Whether to use Generalized Advantage Estimation
            gae_lambda: GAE lambda
            device: Device to use
            seed: Random seed
            rng: Random number generator
        """
        super().__init__(id, device)
        self.id = id
        self.envs = envs
        self.num_envs = envs.num_envs
        self.networks = networks
        self.device = device
        self.seed = seed
        if rng is not None:
            self.np_random = rng
        else:
            self.np_random = np.random.default_rng(self.seed)

        # PPO Parameters
        self.steps_per_iteration = steps_per_iteration
        self.np_weights = weights
        self.weights = th.from_numpy(weights).to(self.device)
        self.batch_size = int(self.num_envs * self.steps_per_iteration)
        self.num_minibatches = num_minibatches
        self.minibatch_size = int(self.batch_size // num_minibatches)
        self.update_epochs = update_epochs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.anneal_lr = anneal_lr
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.norm_adv = norm_adv
        self.target_kl = target_kl
        self.clip_vloss = clip_vloss
        self.gae_lambda = gae_lambda
        self.log = log
        self.gae = gae

        self.optimizer = optim.Adam(networks.parameters(), lr=self.learning_rate, eps=1e-5)

        # Storage setup (the batch)
        self.batch = PPOReplayBuffer(
            self.steps_per_iteration,
            self.num_envs,
            self.networks.obs_shape,
            self.networks.action_shape,
            self.networks.reward_dim,
            self.device,
        )

    def __deepcopy__(self, memo):
        """Deepcopy method.

        Useful for genetic algorithms stuffs.
        """
        copied_net = deepcopy(self.networks)
        copied = type(self)(
            self.id,
            copied_net,
            self.weights.detach().cpu().numpy(),
            self.envs,
            self.log,
            self.steps_per_iteration,
            self.num_minibatches,
            self.update_epochs,
            self.learning_rate,
            self.gamma,
            self.anneal_lr,
            self.clip_coef,
            self.ent_coef,
            self.vf_coef,
            self.clip_vloss,
            self.max_grad_norm,
            self.norm_adv,
            self.target_kl,
            self.gae,
            self.gae_lambda,
            self.device,
        )

        copied.global_step = self.global_step
        copied.optimizer = optim.Adam(copied_net.parameters(), lr=self.learning_rate, eps=1e-5)
        copied.batch = deepcopy(self.batch)
        return copied

    def change_weights(self, new_weights: np.ndarray):
        """Change the weights of the scalarization function.

        Args:
            new_weights: New weights to apply.
        """
        self.weights = th.from_numpy(deepcopy(new_weights)).to(self.device)

    def __extend_to_reward_dim(self, tensor: th.Tensor):
        # This allows to broadcast the tensor to match the additional dimension of rewards
        return tensor.unsqueeze(1).repeat(1, self.networks.reward_dim)

    def __collect_samples(self, obs: th.Tensor, done: th.Tensor):
        """Fills the batch with {self.steps_per_iteration} samples collected from the environments.

        Args:
            obs: current observations
            done: current dones

        Returns:
            next observation and dones
        """
        for step in range(0, self.steps_per_iteration):
            self.global_step += 1 * self.num_envs
            # Compute best action
            with th.no_grad():
                action, logprob, _, value = self.networks.get_action_and_value(obs)
                value = value.view(self.num_envs, self.networks.reward_dim)

            # Perform action on the environment
            next_obs, reward, next_terminated, next_truncated, info = self.envs.step(action.cpu().numpy())
            reward = th.tensor(reward).to(self.device).view(self.num_envs, self.networks.reward_dim)
            # storing to batch
            self.batch.add(obs, action, logprob, reward, done, value)

            # Next iteration
            obs, done = th.Tensor(next_obs).to(self.device), th.Tensor(next_terminated).to(self.device)

            # Episode info logging
            if self.log and "episode" in info.keys():
                indices = np.where(next_terminated | next_truncated)[0]
                for idx in indices:
                    # Reconstructs the dict by extracting the relevant information for each vectorized env
                    info_log = {k: v[idx] for k, v in info["episode"].items()}

                    log_episode_info(
                        info_log,
                        scalarization=np.dot,
                        weights=self.weights,
                        global_timestep=self.global_step,
                        id=self.id,
                    )

        return obs, done

    def __compute_advantages(self, next_obs, next_done):
        """Computes the advantages by replaying experiences from the buffer in reverse.

        Returns:
            MO returns, scalarized advantages
        """
        with th.no_grad():
            next_value = self.networks.get_value(next_obs).reshape(self.num_envs, -1)
            if self.gae:
                advantages = th.zeros_like(self.batch.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.steps_per_iteration)):
                    if t == self.steps_per_iteration - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        _, _, _, _, done_t1, value_t1 = self.batch.get(t + 1)
                        nextnonterminal = 1.0 - done_t1
                        nextvalues = value_t1

                    nextnonterminal = self.__extend_to_reward_dim(nextnonterminal)
                    _, _, _, reward_t, _, value_t = self.batch.get(t)
                    delta = reward_t + self.gamma * nextvalues * nextnonterminal - value_t
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.batch.values
            else:
                returns = th.zeros_like(self.batch.rewards).to(self.device)
                for t in reversed(range(self.steps_per_iteration)):
                    if t == self.steps_per_iteration - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        _, _, _, _, done_t1, _ = self.batch.get(t + 1)
                        nextnonterminal = 1.0 - done_t1
                        next_return = returns[t + 1]

                    nextnonterminal = self.__extend_to_reward_dim(nextnonterminal)
                    _, _, _, reward_t, _, _ = self.batch.get(t)
                    returns[t] = reward_t + self.gamma * nextnonterminal * next_return
                advantages = returns - self.batch.values

        # Scalarization of the advantages (weighted sum)
        advantages = advantages @ self.weights
        return returns, advantages

    @override
    def eval(self, obs: np.ndarray, w):
        """Returns the best action to perform for the given obs

        Returns:
            action as a numpy array (continuous actions)
        """
        obs = th.as_tensor(obs).float().to(self.device)
        obs = obs.unsqueeze(0).repeat(self.num_envs, 1)  # duplicate observation to fit the NN input
        with th.no_grad():
            action, _, _, _ = self.networks.get_action_and_value(obs)

        return action[0].detach().cpu().numpy()

    @override
    def update(self):
        # flatten the batch (b == batch)
        obs, actions, logprobs, _, _, values = self.batch.get_all()
        b_obs = obs.reshape((-1,) + self.networks.obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.networks.action_shape)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1, self.networks.reward_dim)
        b_values = values.reshape(-1, self.networks.reward_dim)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        # Perform multiple passes on the batch (that is shuffled every time)
        for epoch in range(self.update_epochs):
            self.np_random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                # mb == minibatch
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.networks.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with th.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1, self.networks.reward_dim)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + th.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.networks.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # record rewards for plotting purposes
        if self.log:
            wandb.log(
                {
                    f"charts_{self.id}/learning_rate": self.optimizer.param_groups[0]["lr"],
                    f"losses_{self.id}/value_loss": v_loss.item(),
                    f"losses_{self.id}/policy_loss": pg_loss.item(),
                    f"losses_{self.id}/entropy": entropy_loss.item(),
                    f"losses_{self.id}/old_approx_kl": old_approx_kl.item(),
                    f"losses_{self.id}/approx_kl": approx_kl.item(),
                    f"losses_{self.id}/clipfrac": np.mean(clipfracs),
                    f"losses_{self.id}/explained_variance": explained_var,
                    "global_step": self.global_step,
                },
            )

    def train(self, start_time, current_iteration: int, max_iterations: int):
        """A training iteration: trains MOPPO for self.steps_per_iteration * self.num_envs.

        Args:
            start_time: time.time() when the training started
            current_iteration: current iteration number
            max_iterations: maximum number of iterations
        """
        next_obs, _ = self.envs.reset(seed=self.seed)
        next_obs = th.Tensor(next_obs).to(self.device)  # num_envs x obs
        next_done = th.zeros(self.num_envs).to(self.device)

        # Annealing the rate if instructed to do so.
        if self.anneal_lr:
            frac = 1.0 - (current_iteration - 1.0) / max_iterations
            lrnow = frac * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

        # Fills buffer
        next_obs, next_done = self.__collect_samples(next_obs, next_done)

        # Compute advantage on collected samples
        self.returns, self.advantages = self.__compute_advantages(next_obs, next_done)

        # Update neural networks from batch
        self.update()

        # Logging
        print("SPS:", int(self.global_step / (time.time() - start_time)))
        if self.log:
            print(f"Worker {self.id} - Global step: {self.global_step}")
            wandb.log(
                {"charts/SPS": int(self.global_step / (time.time() - start_time)), "global_step": self.global_step},
            )

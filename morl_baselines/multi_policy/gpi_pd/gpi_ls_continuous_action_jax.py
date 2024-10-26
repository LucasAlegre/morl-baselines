"""GPI-LS algorithm with continuous actions in Jax."""

import os
import random
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
import wandb
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from jax.nn import initializers

from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.evaluation import (
    log_all_multi_policy_metrics,
    log_episode_info,
    policy_evaluation_mo,
)
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.prioritized_buffer import PrioritizedReplayBuffer
from morl_baselines.common.utils import unique_tol
from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport


class ActorTrainState(TrainState):
    """Train state for the actor."""

    batch_stats: flax.core.FrozenDict


class RLTrainState(TrainState):
    """Train state for the critic."""

    target_params: flax.core.FrozenDict
    batch_stats: flax.core.FrozenDict
    target_batch_stats: flax.core.FrozenDict


class Policy(nn.Module):
    """MO Policy."""

    action_dim: int
    batch_norm_momentum: float
    action_scale: jnp.array
    action_offset: jnp.array
    policy_noise: float
    noise_clip: float
    num_hidden_layers: int = 2
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray, w: jnp.ndarray, train: bool, add_noise: bool = False, key=None):
        """Forward pass."""
        h = jnp.concatenate([obs, w], axis=-1)
        h = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(h)
        for _ in range(self.num_hidden_layers):
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.leaky_relu(h)
            h = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(h)

        action = nn.Dense(self.action_dim)(h)
        action = jnp.tanh(action)

        if add_noise:
            noise = jax.random.normal(key, action.shape, dtype=jnp.float32) * self.policy_noise
            noise = jnp.clip(noise, -self.noise_clip, self.noise_clip)
            action = jnp.clip(action + noise, -1.0, 1.0)

        return action * self.action_scale + self.action_offset


class QNetwork(nn.Module):
    """MO QNetwork."""

    action_dim: int
    rew_dim: int
    batch_norm_momentum: float
    dropout_rate: Optional[float] = 0.01
    num_hidden_layers: int = 4
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, w: jnp.ndarray, deterministic: bool, train: bool):
        """Forward pass."""
        h = jnp.concatenate([obs, action, w], axis=-1)
        h = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(h)

        for _ in range(self.num_hidden_layers):
            h = nn.Dense(self.hidden_dim)(h)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
            h = nn.leaky_relu(h)
            h = BatchRenorm(use_running_average=not train, momentum=self.batch_norm_momentum)(h)
        h = nn.Dense(self.rew_dim)(h)

        return h


class VectorQNetwork(nn.Module):
    """Vectorized QNetwork."""

    action_dim: int
    rew_dim: int
    batch_norm_momentum: float
    dropout_rate: Optional[float] = 0.01
    n_critics: int = 2
    num_hidden_layers: int = 4
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, w: jnp.ndarray, deterministic: bool, train: bool):
        """Vectorized forward pass."""
        vmap_critic = nn.vmap(
            QNetwork,
            variable_axes={"params": 0, "batch_stats": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "batch_stats": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            action_dim=self.action_dim,
            rew_dim=self.rew_dim,
            batch_norm_momentum=self.batch_norm_momentum,
            dropout_rate=self.dropout_rate,
            num_hidden_layers=self.num_hidden_layers,
            hidden_dim=self.hidden_dim,
        )(obs, action, w, deterministic, train)
        return q_values.reshape((self.n_critics, -1, self.rew_dim))


class GPILSContinuousAction(MOAgent, MOPolicy):
    """GPI-LS algorithm with continuous actions in Jax.

    This version is based on the CrossQ algorithm instead of TD3, and written on Jax for efficiency.

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
        buffer_size: int = 400000,
        net_arch: List = [256, 256],
        batch_size: int = 128,
        num_q_nets: int = 2,
        dropout_rate: Optional[float] = 0.01,
        learning_starts: int = 100,
        gradient_updates: int = 10,
        use_gpi: bool = False,  # In the continuous action case, GPI is only used to selected weights.
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        per: bool = True,
        min_priority: float = 0.1,
        alpha: float = 0.6,
        batch_norm_momentum: float = 0.99,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "GPI-LS Continuous Action - Jax",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
    ):
        """GPI-LS algorithm with continuous actions.

        It extends the CrossQ algorithm to multi-objective RL using GPI-LS.
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
            learning_starts (int, optional): The number of steps to take before starting to train. Defaults to 100.
            gradient_updates (int, optional): The number of gradient steps to take per update. Defaults to 1.
            use_gpi (bool, optional): Whether to use GPI for selecting actions. Defaults to True.
            policy_noise (float, optional): The noise to add to the policy. Defaults to 0.2.
            dropout_rate (float, optional): The dropout rate. Defaults to 0.01.
            noise_clip (float, optional): The noise clipping value. Defaults to 0.5.
            per (bool, optional): Whether to use prioritized experience replay. Defaults to False.
            min_priority (float, optional): The minimum priority to use for prioritized experience replay. Defaults to 0.1.
            alpha (float, optional): The alpha value for prioritized experience replay. Defaults to 0.6.
            batch_norm_momentum (float): Value of momentum for batch renorm.
            project_name (str, optional): The name of the project. Defaults to "MORL Baselines".
            experiment_name (str, optional): The name of the experiment. Defaults to "GPI-PD Continuous Action".
            wandb_entity (Optional[str], optional): The wandb entity. Defaults to None.
            log (bool, optional): Whether to log to wandb. Defaults to True.
            seed (Optional[int], optional): The seed to use. Defaults to None.
        """
        MOAgent.__init__(self, env, device=None, seed=seed)
        MOPolicy.__init__(self, device=None)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.use_gpi = use_gpi
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.buffer_size = buffer_size
        self.num_q_nets = num_q_nets
        self.dropout_rate = dropout_rate
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates
        self.batch_norm_momentum = batch_norm_momentum
        self.per = per
        self.min_priority = min_priority
        self.alpha = alpha
        self.include_w = False
        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.observation_shape, self.action_dim, rew_dim=self.reward_dim, max_size=buffer_size
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.observation_shape, self.action_dim, rew_dim=self.reward_dim, max_size=buffer_size
            )

        key = jax.random.PRNGKey(seed)
        self.key, q_key, actor_key, bn_key, drop_key = jax.random.split(key, 5)

        obs = env.observation_space.sample()
        action = env.action_space.sample()
        w = np.zeros(self.reward_dim, dtype=np.float32)
        action_high = env.action_space.high
        action_low = env.action_space.low
        action_scale = (action_high - action_low) / 2
        action_offset = (action_high + action_low) / 2

        self.actor = Policy(
            self.action_dim,
            action_scale=action_scale,
            action_offset=action_offset,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            num_hidden_layers=len(net_arch),
            hidden_dim=net_arch[0],
            batch_norm_momentum=self.batch_norm_momentum,
        )
        actor_init_variables = self.actor.init({"params": actor_key, "batch_stats": bn_key}, obs, w, train=False)
        self.actor_state = ActorTrainState.create(
            apply_fn=self.actor.apply,
            params=actor_init_variables["params"],
            batch_stats=actor_init_variables["batch_stats"],
            tx=optax.adam(learning_rate=self.learning_rate, b1=0.5),  # CrossQ uses b1=0.5
        )

        self.q_net = VectorQNetwork(
            self.action_dim,
            rew_dim=self.reward_dim,
            batch_norm_momentum=batch_norm_momentum,
            dropout_rate=self.dropout_rate,
            n_critics=self.num_q_nets,
            num_hidden_layers=len(self.net_arch),
            hidden_dim=self.net_arch[0] * 2,  # Wider critic as in CrossQ
        )
        q_init_variables = self.q_net.init(
            {"params": q_key, "batch_stats": bn_key, "dropout": drop_key}, obs, action, w, deterministic=False, train=False
        )
        target_q_init_variables = self.q_net.init(
            {"params": q_key, "batch_stats": bn_key, "dropout": drop_key}, obs, action, w, deterministic=False, train=False
        )
        self.q_state = RLTrainState.create(
            apply_fn=self.q_net.apply,
            params=q_init_variables["params"],
            batch_stats=q_init_variables["batch_stats"],
            target_params=target_q_init_variables["params"],
            target_batch_stats=target_q_init_variables["batch_stats"],
            tx=optax.adam(learning_rate=self.learning_rate, b1=0.5),
        )
        self.q_net.apply = jax.jit(self.q_net.apply, static_argnames=("batch_norm_momentum", "dropout_rate", "deterministic"))
        self.actor.apply = jax.jit(
            self.actor.apply, static_argnames=("batch_norm_momentum", "action_scale", "action_offset", "add_noise", "policy_noise", "noise_clip")
        )

        self.weight_support = []

        self._n_updates = 0

        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def get_config(self):
        """Returns the agent's config."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "use_gpi": self.use_gpi,
            "batch_size": self.batch_size,
            "per": self.per,
            "alpha_per": self.alpha,
            "min_priority": self.min_priority,
            "num_q_nets": self.num_q_nets,
            "gamma": self.gamma,
            "policy_noise": self.policy_noise,
            "noise_clip": self.noise_clip,
            "net_arch": self.net_arch,
            "gradient_updates": self.gradient_updates,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "dropout_rate": self.dropout_rate,
            "batch_norm_momentum": self.batch_norm_momentum,
        }

    def save(self, save_dir="weights/", filename=None):
        """Save the agent's parameters to disk."""
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        saved_params = {}
        saved_params["q_net_state"] = self.q_state
        saved_params["actor_state"] = self.actor_state
        saved_params["M"] = self.weight_support

        filename = self.experiment_name if filename is None else filename
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(saved_params)
        orbax_checkpointer.save(save_dir + filename, saved_params, save_args=save_args, force=True)

    def load(self, path):
        """Load agent's parameters from the given path."""
        target = {"q_net_state": self.q_state, "actor_state": self.actor_state, "M": self.weight_support}
        ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
        restored = ckptr.restore(path, item=None)

        target["M"] = restored["M"]  # for some reason I need to do this

        restored = ckptr.restore(
            path, item=target, restore_args=flax.training.orbax_utils.restore_args_from_target(target, mesh=None)
        )

        self.q_state = restored["q_net_state"]
        self.actor_state = restored["actor_state"]
        self.weight_support = [w for w in restored["M"].values()]

    def sample_batch_experiences(self):
        """Samples a mini-batch of experiences."""
        return self.replay_buffer.sample(self.batch_size)

    @staticmethod
    @partial(jax.jit, static_argnames=["gamma", "kappa"])
    def update_critic(q_state, actor_state, w, obs, actions, rewards, next_obs, dones, kappa, gamma, key):
        """Updates the agent's critic."""
        key, noise_key, inds_key, drop_key = jax.random.split(key, 4)

        next_actions = actor_state.apply_fn(
            {"params": actor_state.params, "batch_stats": actor_state.batch_stats},
            next_obs,
            w,
            train=False,
            add_noise=True,
            key=noise_key,
        )

        def mse_loss(params, batch_stats, drop_key):
            catted_q_values, state_updates = q_state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                jnp.concatenate([obs, next_obs], axis=0),
                jnp.concatenate([actions, next_actions], axis=0),
                jnp.concatenate([w, w], axis=0),
                mutable=["batch_stats"],
                rngs={"dropout": drop_key},
                deterministic=False,
                train=True,
            )
            current_mo_qvalues, next_mo_qvalues = jnp.split(catted_q_values, 2, axis=1)
            if next_mo_qvalues.shape[0] > 2:
                next_mo_qvalues = jax.random.choice(inds_key, next_mo_qvalues, (2,), replace=False, axis=0)
            next_q_values = (next_mo_qvalues * w).sum(axis=2)

            min_ind = next_q_values.argmin(axis=0)
            next_mo_qvalues = jnp.take_along_axis(next_mo_qvalues, min_ind[None, ..., None], axis=0).squeeze(0)
            target = rewards + (1 - dones) * gamma * next_mo_qvalues
            tds = current_mo_qvalues - jax.lax.stop_gradient(target)
            loss = jnp.abs(tds)
            loss = jnp.where(loss < kappa, 0.5 * loss**2, loss * kappa).mean()
            return loss, (state_updates, tds)

        (loss_value, (state_updates, td_error)), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            q_state.params, q_state.batch_stats, drop_key
        )
        q_state = q_state.apply_gradients(grads=grads)
        q_state = q_state.replace(batch_stats=state_updates["batch_stats"])

        return q_state, loss_value, td_error, key

    @staticmethod
    @jax.jit
    def update_actor(actor_state, q_state, obs, w, key):
        """Updates the agent's actor."""
        key, drop_key = jax.random.split(key)

        def actor_loss(params, batch_stats, drop_key):
            actions, state_updates = actor_state.apply_fn(
                {"params": params, "batch_stats": batch_stats}, obs, w, mutable=["batch_stats"], train=True
            )
            mo_q = q_state.apply_fn(
                {"params": q_state.params, "batch_stats": q_state.batch_stats},
                obs,
                actions,
                w,
                rngs={"dropout": drop_key},
                deterministic=False,
                train=False,
            )
            q = (mo_q * w).sum(axis=2)
            loss = -q.mean()
            return loss, state_updates

        (actor_loss_value, state_updates), grads = jax.value_and_grad(actor_loss, has_aux=True)(
            actor_state.params, actor_state.batch_stats, drop_key
        )
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(batch_stats=state_updates["batch_stats"])
        return actor_state, actor_loss_value, key

    def update(self, weight):
        """Updates the agent's parameters."""
        critic_losses = []
        actor_losses = []
        for g in range(self.gradient_updates):
            if self.per:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones, idxes = self.sample_batch_experiences()
            else:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = self.sample_batch_experiences()

            if len(self.weight_support) > 1:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = (
                    np.vstack([s_obs] * 2),
                    np.vstack([s_actions] * 2),
                    np.vstack([s_rewards] * 2),
                    np.vstack([s_next_obs] * 2),
                    np.vstack([s_dones] * 2),
                )
                w = np.vstack(
                    [weight for _ in range(s_obs.shape[0] // 2)] + random.choices(self.weight_support, k=s_obs.shape[0] // 2)
                )
            else:
                w = weight.repeat(s_obs.shape[0], 1)

            self.q_state, loss, td_error, self.key = GPILSContinuousAction.update_critic(
                self.q_state,
                self.actor_state,
                w,
                s_obs,
                s_actions,
                s_rewards,
                s_next_obs,
                s_dones,
                self.min_priority,
                self.gamma,
                self.key,
            )
            self._n_updates += 1

            critic_losses.append(loss.item())

            if self.per:
                td_error = jax.device_get(td_error)
                td_error = np.abs((td_error[:, : len(idxes)] * w[: len(idxes)]).sum(axis=2))
                per = np.max(td_error, axis=0)
                priority = per.clip(min=self.min_priority) ** self.alpha
                self.replay_buffer.update_priorities(idxes, priority)

        self.actor_state, actor_loss, self.key = GPILSContinuousAction.update_actor(
            self.actor_state, self.q_state, s_obs, w, self.key
        )
        actor_losses.append(actor_loss.item())

        if self.log and self.global_step % 100 == 0:
            if self.per:
                wandb.log(
                    {
                        "metrics/mean_priority": np.mean(priority),
                        "metrics/max_priority": np.max(priority),
                        "metrics/mean_td_error_w": np.mean(per),
                    },
                    commit=False,
                )
            wandb.log(
                {
                    "losses/critic_loss": np.mean(critic_losses),
                    "losses/actor_loss": np.mean(actor_losses),
                    "global_step": self.global_step,
                }
            )

    @staticmethod
    @jax.jit
    def gpi_action(actor_state, q_state, obs, w, M):
        """GPI with continuous actions."""
        M = jnp.vstack(M)

        obs_m = jnp.repeat(obs.reshape(1, -1), M.shape[0], axis=0)
        actions_per_policy = actor_state.apply_fn(
            {"params": actor_state.params, "batch_stats": actor_state.batch_stats}, obs_m, M, train=False
        )

        def maxqpi(obs, action, w):
            obs = jnp.repeat(obs.reshape(1, -1), M.shape[0], axis=0)
            action = jnp.repeat(action.reshape(1, -1), M.shape[0], axis=0)
            mo_q_values = q_state.apply_fn(
                {"params": q_state.params, "batch_stats": q_state.batch_stats},
                obs,
                action,
                M,
                deterministic=True,
                train=False,
            )
            q_values = (mo_q_values * w).sum(axis=2)  # (n_critics, |M|)
            q_values = q_values.mean(axis=0)  # |M|
            max_q = q_values.max()
            return max_q

        vmaxqpi = jax.vmap(maxqpi, in_axes=(None, 0, None))

        q_values = vmaxqpi(obs, actions_per_policy, w)
        max_a = q_values.argmax()
        action = actions_per_policy[max_a]
        action = jax.device_get(action)
        return action

    def set_weight_support(self, M: List[np.ndarray]):
        """Set the weight support set."""
        self.weight_support = M.copy()

    def eval(self, obs: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Evaluate policy for the given observation and weight vector."""
        if self.include_w:
            self.weight_support.append(w)

        if self.use_gpi:
            action = GPILSContinuousAction.gpi_action(
                self.actor_state,
                self.q_state,
                obs,
                w,
                self.weight_support,
            )
        else:
            action = GPILSContinuousAction.max_action(self.actor_state, obs, w)

        if self.include_w:
            self.weight_support.pop(-1)

        action = jax.device_get(action)
        return action

    def act(self, obs, w) -> int:
        """Act with exploration noise."""
        self.key, noise_key = jax.random.split(self.key)
        action = self.actor_state.apply_fn(
            {"params": self.actor_state.params, "batch_stats": self.actor_state.batch_stats},
            obs,
            w,
            train=False,
            add_noise=True,
            key=noise_key,
        )
        action = jax.device_get(action)
        return action

    @staticmethod
    @jax.jit
    def max_action(actor_state, obs, w) -> np.ndarray:
        """Returns action given directly by the policy."""
        action = actor_state.apply_fn(
            {"params": actor_state.params, "batch_stats": actor_state.batch_stats}, obs, w, train=False
        )
        action = jax.device_get(action)
        return action

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

        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes

        obs, info = self.env.reset()
        for _ in range(1, total_timesteps + 1):
            self.global_step += 1

            if self.global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.act(obs, weight)

            action_env = action

            next_obs, vector_reward, terminated, truncated, info = self.env.step(action_env)

            self.replay_buffer.add(obs, action, vector_reward, next_obs, terminated)

            if self.global_step >= self.learning_starts:
                self.update(weight)

            if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                self.policy_eval(eval_env, weights=weight, log=self.log)

            if terminated or truncated:
                obs, info = self.env.reset()
                self.num_episodes += 1

                if self.log and "episode" in info.keys():
                    log_episode_info(info["episode"], np.dot, weight, self.global_step)

                if change_weight_every_episode:
                    weight = random.choice(weight_support)
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
                self.save(filename=f"GPI-LS Jax iter={iter}")

        self.close_wandb()


PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Axes = Union[int, Sequence[int]]


class BatchRenorm(Module):
    """BatchRenorm Module (https://arxiv.org/abs/1702.03275).

    This implementation is from https://github.com/araffin/sbx/blob/master/sbx/common/jax_layers.py.

    BatchRenorm is an improved version of vanilla BatchNorm. Contrary to BatchNorm,
    BatchRenorm uses the running statistics for normalizing the batches after a warmup phase.
    This makes it less prone to suffer from "outlier" batches that can happen
    during very long training runs and, therefore, is more robust during long training runs.

    During the warmup phase, it behaves exactly like a BatchNorm layer.

    Usage Note:
    If we define a model with BatchRenorm, for example::

      BRN = BatchRenorm(use_running_average=False, momentum=0.99, epsilon=0.001, dtype=jnp.float32)

    The initialized variables dict will contain in addition to a 'params'
    collection a separate 'batch_stats' collection that will contain all the
    running statistics for all the BatchRenorm layers in a model::

      vars_initialized = BRN.init(key, x)  # {'params': ..., 'batch_stats': ...}

    We then update the batch_stats during training by specifying that the
    `batch_stats` collection is mutable in the `apply` method for our module.::

      vars_in = {'params': params, 'batch_stats': old_batch_stats}
      y, mutated_vars = BRN.apply(vars_in, x, mutable=['batch_stats'])
      new_batch_stats = mutated_vars['batch_stats']

    During eval we would define BRN with `use_running_average=True` and use the
    batch_stats collection from training to set the statistics.  In this case
    we are not mutating the batch statistics collection, and needn't mark it
    mutable::

      vars_in = {'params': params, 'batch_stats': training_batch_stats}
      y = BRN.apply(vars_in, x)

    Attributes:
      use_running_average: if True, the statistics stored in batch_stats will be
        used. Else the running statistics will be first updated and then used to normalize.
      axis: the feature or non-batch axis of the input.
      momentum: decay rate for the exponential moving average of the batch
        statistics.
      epsilon: a small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_bias:  if True, bias (beta) is added.
      use_scale: if True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: initializer for bias, by default, zero.
      scale_init: initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
        examples on the first two and last two devices. See `jax.lax.psum` for
        more details.
      use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
    """

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 0.001
    warmup_steps: int = 100_000
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    # This parameter was added in flax.linen 0.7.2 (08/2023)
    # commented out to be compatible with a wider range of jax versions
    # TODO: re-activate in some months (04/2024)
    use_fast_variance: bool = False

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """Normalizes the input using batch statistics.

        During initialization (when `self.is_initializing()` is `True`) the running
        average of the batch statistics will not be updated. Therefore, the inputs
        fed during initialization don't need to match that of the actual input
        distribution and the reduction axis (set with `axis_name`) does not have
        to exist.

        Args:
          x: the input to be normalized.
          use_running_average: if true, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.

        Returns:
          Normalized inputs (the same shape as inputs).
        """
        use_running_average = merge_param("use_running_average", self.use_running_average, use_running_average)
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            "batch_stats",
            "mean",
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable("batch_stats", "var", lambda s: jnp.ones(s, jnp.float32), feature_shape)

        r_max = self.variable(
            "batch_stats",
            "r_max",
            lambda s: s,
            3,
        )
        d_max = self.variable(
            "batch_stats",
            "d_max",
            lambda s: s,
            5,
        )
        steps = self.variable(
            "batch_stats",
            "steps",
            lambda s: s,
            0,
        )

        if use_running_average:
            custom_mean = ra_mean.value
            custom_var = ra_var.value
        else:
            batch_mean, batch_var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )

            if self.is_initializing():
                custom_mean = batch_mean
                custom_var = batch_var
            else:
                std = jnp.sqrt(batch_var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                # scale
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                # bias
                d = jax.lax.stop_gradient((batch_mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)

                # BatchNorm normalization, using minibatch stats and running average stats
                # Because we use _normalize, this is equivalent to
                # ((x - x_mean) / sigma) * r + d = ((x - x_mean) * r + d * sigma) / sigma
                # where sigma = sqrt(var)
                affine_mean = batch_mean - d * jnp.sqrt(batch_var + self.epsilon) / r
                affine_var = (batch_var + self.epsilon) / (r**2)

                # Note: in the original paper, after some warmup phase (batch norm phase of 5k steps)
                # the constraints are linearly relaxed to r_max/d_max over 40k steps
                # Here we only have a warmup phase
                is_warmed_up = jnp.greater_equal(steps.value, self.warmup_steps).astype(jnp.float32)
                custom_mean = is_warmed_up * affine_mean + (1.0 - is_warmed_up) * batch_mean
                custom_var = is_warmed_up * affine_var + (1.0 - is_warmed_up) * batch_var

                ra_mean.value = self.momentum * ra_mean.value + (1.0 - self.momentum) * batch_mean
                ra_var.value = self.momentum * ra_var.value + (1.0 - self.momentum) * batch_var
                steps.value += 1

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )

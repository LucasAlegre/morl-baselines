"""GPI-LS algorithm in Jax."""
import os
import random
from functools import partial
from typing import List, Optional

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

# flax.config.update('flax_use_orbax_checkpointing', True)
import orbax
import wandb
from flax.training import checkpoints, orbax_utils
from flax.training.train_state import TrainState

from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.evaluation import (
    log_all_multi_policy_metrics,
    log_episode_info,
    policy_evaluation_mo,
)
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.prioritized_buffer import PrioritizedReplayBuffer
from morl_baselines.common.utils import linearly_decaying_value, unique_tol
from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport


class QNetwork(nn.Module):
    """Multi-Objective Q Network."""

    action_dim: int
    rew_dim: int
    dropout_rate: Optional[float] = 0.01
    use_layer_norm: bool = True
    num_hidden_layers: int = 4
    hidden_dim: int = 256
    image_obs: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, w: jnp.ndarray, deterministic: bool):
        """Forward pass of the Q network."""
        if self.image_obs:
            if len(obs.shape) == 3:
                obs = obs[None]
            x = jnp.transpose(obs, (0, 2, 3, 1))
            x = x / (255.0)
            x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding=0)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding=0)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding=0)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))
            x = nn.Dense(self.hidden_dim)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            h_obs = nn.relu(x)
        else:
            h_obs = nn.Dense(self.hidden_dim)(obs)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                h_obs = nn.Dropout(rate=self.dropout_rate)(h_obs, deterministic=deterministic)
            if self.use_layer_norm:
                h_obs = nn.LayerNorm()(h_obs)
            h_obs = nn.relu(h_obs)

        h_w = nn.Dense(self.hidden_dim)(w)
        if self.dropout_rate is not None and self.dropout_rate > 0:
            h_w = nn.Dropout(rate=self.dropout_rate)(h_w, deterministic=deterministic)
        if self.use_layer_norm:
            h_w = nn.LayerNorm()(h_w)
        h_w = nn.relu(h_w)

        h = h_obs * h_w
        for _ in range(self.num_hidden_layers - 1):
            h = nn.Dense(self.hidden_dim)(h)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
            if self.use_layer_norm:
                h = nn.LayerNorm()(h)
            h = nn.relu(h)
        x = nn.Dense(self.action_dim * self.rew_dim)(h)
        return x


class VectorQNetwork(nn.Module):
    action_dim: int
    rew_dim: int
    use_layer_norm: bool = True
    dropout_rate: Optional[float] = 0.01
    n_critics: int = 2
    num_hidden_layers: int = 4
    hidden_dim: int = 256
    image_obs: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray, w: jnp.ndarray, deterministic: bool):
        """Forward pass of the Q network."""
        vmap_critic = nn.vmap(
            QNetwork,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            action_dim=self.action_dim,
            rew_dim=self.rew_dim,
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.use_layer_norm,
            num_hidden_layers=self.num_hidden_layers,
            hidden_dim=self.hidden_dim,
            image_obs=self.image_obs,
        )(obs, w, deterministic)
        return q_values.reshape((self.n_critics, -1, self.action_dim, self.rew_dim))


class TrainState(TrainState):
    """Train state for the Q network."""
    target_params: flax.core.FrozenDict


class GPILS(MOAgent, MOPolicy):
    """GPI-LS Algorithm.

    Sample-Efficient Multi-Objective Learning via Generalized Policy Improvement Prioritization
    Lucas N. Alegre, Ana L. C. Bazzan, Diederik M. Roijers, Ann Nowé, Bruno C. da Silva
    AAMAS 2023
    Paper: https://arxiv.org/abs/2301.07784
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
        net_arch: List = [256, 256, 256, 256],
        num_nets: int = 2,
        batch_size: int = 128,
        learning_starts: int = 100,
        gradient_updates: int = 20,
        gamma: float = 0.99,
        use_gpi: bool = True,
        gpi_type: str = "gpi",
        pessimism: float = 0.0,
        per: bool = False,
        alpha_per: float = 0.6,
        min_priority: float = 0.01,
        drop_rate: float = 0.01,
        layer_norm: bool = True,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "GPI-LS - Jax",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        device=None,
    ):
        """Initialize the GPI-LS algorithm.

        Args:
            env: The environment to learn from.
            learning_rate: The learning rate.
            initial_epsilon: The initial epsilon value.
            final_epsilon: The final epsilon value.
            epsilon_decay_steps: The number of steps to decay epsilon.
            tau: The soft update coefficient.
            target_net_update_freq: The target network update frequency.
            buffer_size: The size of the replay buffer.
            net_arch: The network architecture.
            num_nets: The number of networks.
            batch_size: The batch size.
            learning_starts: The number of steps before learning starts.
            gradient_updates: The number of gradient updates per step.
            gamma: The discount factor.
            use_gpi: Whether to use GPI.
            gpi_type: "gpi" or "ugpi" for uncertainty-aware GPI.
            pessimism: Pessimism level when using ugpi.
            per: Whether to use PER.
            alpha_per: The alpha parameter for PER.
            min_priority: The minimum priority for PER.
            drop_rate: The dropout rate.
            layer_norm: Whether to use layer normalization.
            project_name: The name of the project.
            experiment_name: The name of the experiment.
            wandb_entity: The name of the wandb entity.
            log: Whether to log.
            seed: The seed for random number generators.
            device: The device to use.
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device=device)
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.tau = tau
        self.target_net_update_freq = target_net_update_freq
        self.gamma = gamma
        self.use_gpi = use_gpi
        self.gpi_type = gpi_type
        self.pessimism = pessimism
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates
        self.num_nets = num_nets
        self.drop_rate = drop_rate
        self.layer_norm = layer_norm
        self.per = per
        self.include_w = False

        key = jax.random.PRNGKey(seed)
        self.key, psi_key, dropout_key = jax.random.split(key, 3)

        obs = env.observation_space.sample()
        w = np.zeros(self.reward_dim, dtype=np.float32)
        self.image_obs = len(obs.shape) > 2
        self.q_net = VectorQNetwork(
            self.action_dim,
            self.reward_dim,
            self.layer_norm,
            self.drop_rate,
            self.num_nets,
            num_hidden_layers=len(self.net_arch),
            hidden_dim=self.net_arch[0],
            image_obs=self.image_obs,
        )
        self.q_state = TrainState.create(
            apply_fn=self.q_net.apply,
            params=self.q_net.init(
                {"params": psi_key, "dropout": dropout_key},
                obs,
                w,
                deterministic=False,
            ),
            target_params=self.q_net.init(
                {"params": psi_key, "dropout": dropout_key},
                obs,
                w,
                deterministic=False,
            ),
            tx=optax.adam(learning_rate=self.learning_rate),
        )
        self.q_net.apply = jax.jit(self.q_net.apply, static_argnames=("dropout_rate", "use_layer_norm", "deterministic"))

        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.observation_shape, 1, rew_dim=self.reward_dim, max_size=buffer_size, action_dtype=np.uint8
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.observation_shape, 1, rew_dim=self.reward_dim, max_size=buffer_size, action_dtype=np.uint8
            )
        self.min_priority = min_priority
        self.alpha = alpha_per
        self.weight_support = []

        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def get_config(self):
        """Return the configuration of the agent."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps:": self.epsilon_decay_steps,
            "final_epsilon": self.final_epsilon,
            "batch_size": self.batch_size,
            "use_gpi": self.use_gpi,
            "gpi_type": self.gpi_type,
            "per": self.per,
            "alpha_per": self.alpha,
            "min_priority": self.min_priority,
            "tau": self.tau,
            "num_nets": self.num_nets,
            "target_net_update_freq": self.target_net_update_freq,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "gradient_updates": self.gradient_updates,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "drop_rate": self.drop_rate,
            "layer_norm": self.layer_norm,
        }

    def save(self, save_dir="weights/", filename=None):
        """Save the model parameters."""
        if not os.path.isabs(save_dir):
            save_dir = os.path.join(os.getcwd(), save_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        saved_params = {}
        saved_params["q_net_state"] = self.q_state
        saved_params["M"] = self.weight_support

        filename = self.experiment_name if filename is None else filename
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(saved_params)
        orbax_checkpointer.save(save_dir + filename, saved_params, save_args=save_args, force=True)

    def load(self, path, step=None):
        """Load the model parameters."""
        target = {"q_net_state": self.q_state, "M": self.weight_support}
        
        ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
        restored = ckptr.restore(path, item=None)

        target["M"] = restored["M"]  # for some reason I need to do this
        restored = ckptr.restore(path, item=target, restore_args=flax.training.orbax_utils.restore_args_from_target(target, mesh=None))

        self.q_state = restored["q_net_state"]
        self.weight_support = [w for w in restored["M"].values()]
        # self.M = [w for w in restored["M"]]

    def _sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=False, device=self.device)

    @staticmethod
    @partial(jax.jit, static_argnames=["q_net", "gamma", "min_priority"])
    def _update_q(q_net, q_state, w, obs, actions, rewards, next_obs, dones, gamma, min_priority, key):
        key, inds_key, dropout_key_target, dropout_key_current = jax.random.split(key, 4)

        # DroQ update
        if q_net.n_critics >= 2:
            psi_values_next = q_net.apply(
                q_state.target_params, next_obs, w, deterministic=False, rngs={"dropout": dropout_key_target}
            )
            if psi_values_next.shape[0] > 2:
                inds = jax.random.randint(inds_key, (2,), 0, psi_values_next.shape[0])
                psi_values_next = psi_values_next[inds]
            q_values_next = (psi_values_next * w.reshape(w.shape[0], 1, w.shape[1])).sum(axis=3)
            min_inds = q_values_next.argmin(axis=0)
            min_psi_values = jnp.take_along_axis(psi_values_next, min_inds[None, ..., None], 0).squeeze(0)

            max_q = (min_psi_values * w.reshape(w.shape[0], 1, w.shape[1])).sum(axis=2)
            max_acts = max_q.argmax(axis=1)
            target = min_psi_values[jnp.arange(min_psi_values.shape[0]), max_acts]

            def mse_loss(params, droptout_key):
                psi_values = q_net.apply(params, obs, w, deterministic=False, rngs={"dropout": droptout_key})
                psi_values = psi_values[:, jnp.arange(psi_values.shape[1]), actions.squeeze()]
                tds = psi_values - target_psi
                loss = jnp.abs(tds)
                loss = jnp.where(loss < min_priority, 0.5 * loss**2, loss * min_priority)
                return loss.mean(), tds

        # DDQN update
        else:
            psi_values_next = q_net.apply(q_state.target_params, next_obs, w, deterministic=True)[0]
            psi_values_not_target = q_net.apply(q_state.params, next_obs, w, deterministic=True)
            q_values_next = (psi_values_not_target * w.reshape(w.shape[0], 1, w.shape[1])).sum(axis=3)[0]
            max_acts = q_values_next.argmax(axis=1)
            target = psi_values_next[jnp.arange(psi_values_next.shape[0]), max_acts]

            def mse_loss(params, droptout_key):
                psi_values = q_net.apply(params, obs, w, deterministic=True)
                psi_values = psi_values[:, jnp.arange(psi_values.shape[1]), actions.squeeze()]
                tds = psi_values - target_psi
                loss = jnp.abs(tds)
                loss = jnp.where(loss < min_priority, 0.5 * loss**2, loss * min_priority)
                return loss.mean(), tds

        target_psi = rewards + (1 - dones) * gamma * target

        (loss_value, td_error), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params, dropout_key_current)
        q_state = q_state.apply_gradients(grads=grads)

        return q_state, loss_value, td_error, key

    def update(self, weight):
        """Update the parameters of the networks."""
        critic_losses = []
        for g in range(self.gradient_updates):
            if self.per:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones, idxes = self._sample_batch_experiences()
            else:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = self._sample_batch_experiences()

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

            self.q_state, loss, td_error, self.key = GPILS._update_q(
                self.q_net,
                self.q_state,
                w,
                s_obs,
                s_actions,
                s_rewards,
                s_next_obs,
                s_dones,
                self.gamma,
                self.min_priority,
                self.key,
            )
            critic_losses.append(loss.item())

            if self.per:
                td_error = jax.device_get(td_error)
                td_error = np.abs((td_error[:, : len(idxes)] * w[: len(idxes)]).sum(axis=2))
                per = np.max(td_error, axis=0)
                priority = per.clip(min=self.min_priority) ** self.alpha
                self.replay_buffer.update_priorities(idxes, priority)

        if self.tau != 1 or self.global_step % self.target_net_update_freq == 0:
            self.q_state = GPILS._target_net_update(self.q_state)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon, self.epsilon_decay_steps, self.global_step, self.learning_starts, self.final_epsilon
            )

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
                    "metrics/epsilon": self.epsilon,
                    "global_step": self.global_step,
                }
            )

    @staticmethod
    @jax.jit
    def _target_net_update(q_state):
        q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))
        return q_state

    @staticmethod
    @partial(jax.jit, static_argnames=["q_net", "pessimism"])
    def ugpi_action(q_net, q_state, obs, w, M, pessimism, key):
        """Uncertainty-Aware GPI (uGPI)."""
        M = jnp.stack(M)

        obs_m = obs.reshape(1, *obs.shape).repeat(M.shape[0], axis=0)
        psi_values = q_net.apply(q_state.params, obs_m, M, deterministic=True)
        q_values = (psi_values * w.reshape(1, 1, 1, w.shape[0])).sum(axis=3)

        n = q_values.shape[0]
        # tinv(0.9, 9) = 1.383028, tinv(0.95, 9) = 1.833113, tinv(0.99, 9) = 2.821438
        if pessimism == 0.9:
            tinv = 1.383028
        elif pessimism == 0.95:
            tinv = 1.833113
        elif pessimism == 0.99:
            tinv = 2.821438
        # LB = v.mean() - stddev(v) / math.sqrt(n) * tinv(1.0 - delta, n - 1)
        # sqrt(10) = 3.162278
        if pessimism == 1.0:
            q_values = q_values.mean(axis=0) - pessimism * q_values.std(axis=0)
        else:
            q_values = q_values.mean(axis=0) - q_values.std(axis=0) / jnp.sqrt(n) * tinv

        max_q = q_values.max(axis=1)
        policy_index = max_q.argmax()  # max_i max_a q(s,a,w_i)
        action = q_values[policy_index].argmax()

        return action, key

    @staticmethod
    @partial(jax.jit, static_argnames=["q_net"])
    def gpi_action(q_net, q_state, obs, w, M, key):
        """Generalized Policy Improvement (GPI)."""
        M = jnp.stack(M)

        # key, subkey = jax.random.split(key)
        obs_m = obs.reshape(1, *obs.shape).repeat(M.shape[0], axis=0)
        psi_values = q_net.apply(q_state.params, obs_m, M, deterministic=True)
        q_values = (psi_values * w.reshape(1, 1, 1, w.shape[0])).sum(axis=3)

        q_values = q_values.mean(axis=0)

        max_q = q_values.max(axis=1)
        policy_index = max_q.argmax()  # max_i max_a q(s,a,w_i)
        action = q_values[policy_index].argmax()

        return action, key

    def eval(self, obs: np.ndarray, w: np.ndarray) -> int:
        """Evaluate the policy."""
        if type(obs) is gym.wrappers.frame_stack.LazyFrames:
            obs = np.array(obs)

        if self.include_w:
            self.weight_support.append(w)

        if self.use_gpi:
            if self.gpi_type == "ugpi":
                action, self.key = GPILS.ugpi_action(
                    self.q_net, self.q_state, obs, w, self.weight_support, self.pessimism, self.key
                )
            elif self.gpi_type == "gpi":
                action, self.key = GPILS.gpi_action(self.q_net, self.q_state, obs, w, self.weight_support, self.key)
        else:
            action, self.key = GPILS.max_action(self.q_net, self.q_state, obs, w, self.key)

        if self.include_w:
            self.weight_support.pop(-1)

        action = jax.device_get(action)
        return action

    def _act(self, obs, w) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            if type(obs) is gym.wrappers.frame_stack.LazyFrames:
                obs = np.array(obs)
            if self.use_gpi:
                action, self.key = GPILS.gpi_action(self.q_net, self.q_state, obs, w, self.weight_support, self.key)
                action = jax.device_get(action)
            else:
                action, self.key = GPILS.max_action(self.q_net, self.q_state, obs, w, self.key)
                action = jax.device_get(action)
            return action

    @staticmethod
    @partial(jax.jit, static_argnames=["q_net"])
    def max_action(q_net, q_state, obs, w, key) -> int:
        """Select the action with the maximum Q-value."""
        psi_values = q_net.apply(q_state.params, obs, w, deterministic=True)
        q_values = (psi_values * w.reshape(1, w.shape[0])).sum(axis=3)
        q_values = q_values.mean(axis=0).squeeze(0)
        action = q_values.argmax()
        action = jax.device_get(action)
        return action, key

    def set_weight_support(self, M: List[np.ndarray]):
        """Set the weight support set."""
        self.weight_support = M.copy()

    def train_iteration(
        self,
        total_timesteps: int,
        weight: np.ndarray,
        weight_support: List[np.ndarray],
        change_w_every_episode: bool = True,
        reset_num_timesteps: bool = True,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 1000,
        reset_learning_starts: bool = False,
    ):
        """Train the agent for one iteration.

        Args:
            total_timesteps (int): Number of timesteps to train for
            weight (np.ndarray): Weight vector
            weight_support (List[np.ndarray]): Weight support set
            change_w_every_episode (bool): Whether to change the weight vector at the end of each episode
            reset_num_timesteps (bool): Whether to reset the number of timesteps
            eval_env (Optional[gym.Env]): Environment to evaluate on
            eval_freq (int): Number of timesteps between evaluations
            reset_learning_starts (bool): Whether to reset the learning starts
        """
        weight_support = unique_tol(weight_support)  # remove duplicates
        self.set_weight_support(weight_support)

        self.police_indices = []
        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        if reset_learning_starts:  # Resets epsilon-greedy exploration
            self.learning_starts = self.global_step

        obs, info = self.env.reset()
        for _ in range(1, total_timesteps + 1):
            self.global_step += 1

            if self.global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self._act(obs, weight)

            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)

            self.replay_buffer.add(obs, action, vec_reward, next_obs, terminated)

            if self.global_step >= self.learning_starts:
                self.update(weight)

            if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                self.policy_eval(eval_env, weights=weight, log=self.log)

            if terminated or truncated:
                obs, _ = self.env.reset()
                self.num_episodes += 1

                if self.log and "episode" in info.keys():
                    log_episode_info(info["episode"], np.dot, weight, self.global_step)

                if change_w_every_episode:
                    weight = random.choice(weight_support)
            else:
                obs = next_obs

    def train(
        self,
        total_timesteps: int,
        eval_env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        num_eval_weights_for_eval: int = 50,
        timesteps_per_iter: int = 10000,
        weight_selection_algo: str = "gpi-ls",
        eval_freq: int = 1000,
        eval_mo_freq: int = 10000,
        checkpoints: bool = True,
    ):
        """Train agent.

        Args:
            total_timesteps (int): Number of timesteps to train for.
            eval_env (gym.Env): Environment to evaluate on.
            ref_point (np.ndarray): Reference point for hypervolume calculation.
            known_pareto_front (Optional[List[np.ndarray]]): Optimal Pareto front if known.
            num_eval_weights_for_front: Number of weights to evaluate for the Pareto front.
            num_eval_episodes_for_front: number of episodes to run when evaluating the policy.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            timesteps_per_iter (int): Number of timesteps to train for per iteration.
            weight_selection_algo (str): Weight selection algorithm to use.
            eval_freq (int): Number of timesteps between evaluations.
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
                    "timesteps_per_iter": timesteps_per_iter,
                    "weight_selection_algo": weight_selection_algo,
                    "eval_freq": eval_freq,
                    "eval_mo_freq": eval_mo_freq,
                }
            )
        max_iter = total_timesteps // timesteps_per_iter
        linear_support = LinearSupport(num_objectives=self.reward_dim, epsilon=0.0 if weight_selection_algo == "ols" else None)

        weight_history = []

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
            weight_history.append(w)
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
                change_w_every_episode=weight_selection_algo == "gpi-ls",
                eval_env=eval_env,
                eval_freq=eval_freq,
                reset_num_timesteps=False,
                reset_learning_starts=False,
            )

            if weight_selection_algo == "ols":
                value = policy_evaluation_mo(self, eval_env, w, rep=num_eval_episodes_for_front)[3]
                linear_support.add_solution(value, w)
            elif weight_selection_algo == "gpi-ls":
                for wcw in M:
                    n_value = policy_evaluation_mo(self, eval_env, wcw, rep=num_eval_episodes_for_front)[3]
                    linear_support.add_solution(n_value, wcw)

            self.set_weight_support(linear_support.get_weight_support())
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

            if checkpoints:
                self.save(filename=f"GPI-PD {weight_selection_algo} iter={iter}")

        self.close_wandb()

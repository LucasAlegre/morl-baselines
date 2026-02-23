"""Replay buffer for multi-objective reinforcement learning."""

import numpy as np
import torch as th


class ReplayBuffer:
    """Multi-objective replay buffer for multi-objective reinforcement learning.

    Memory is allocated on-demand starting from a small initial capacity and
    doubling until ``max_size`` is reached, rather than pre-allocating the full
    ``max_size`` up front.  Once the buffer is full, oldest transitions are
    overwritten in circular fashion (identical behaviour to the previous
    implementation).
    """

    _GROWTH_FACTOR = 2

    def __init__(
        self,
        obs_shape,
        action_dim,
        rew_dim=1,
        max_size=100000,
        obs_dtype=np.float32,
        action_dtype=np.float32,
        initial_capacity=1024,
    ):
        """Initialize the replay buffer.

        Args:
            obs_shape: Shape of the observations
            action_dim: Dimension of the actions
            rew_dim: Dimension of the rewards
            max_size: Maximum size of the buffer
            obs_dtype: Data type of the observations
            action_dtype: Data type of the actions
            initial_capacity: Number of slots allocated at construction time.
                The buffer grows automatically (doubling each time) up to
                ``max_size``.  Defaults to ``min(1024, max_size)``.
        """
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.rew_dim = rew_dim
        self.max_size = max_size
        self.obs_dtype = obs_dtype
        self.action_dtype = action_dtype

        self._capacity = min(initial_capacity, max_size)
        self.ptr = 0
        self.size = 0

        self._allocate(self._capacity)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _allocate(self, capacity):
        """Allocate (or reallocate) backing arrays for *capacity* slots.

        Existing data up to ``self.size`` is preserved.
        """
        new_obs = np.empty((capacity,) + self.obs_shape, dtype=self.obs_dtype)
        new_next_obs = np.empty((capacity,) + self.obs_shape, dtype=self.obs_dtype)
        new_actions = np.empty((capacity, self.action_dim), dtype=self.action_dtype)
        new_rewards = np.empty((capacity, self.rew_dim), dtype=np.float32)
        new_dones = np.empty((capacity, 1), dtype=np.float32)

        if self.size > 0:
            n = self.size
            new_obs[:n] = self.obs[:n]
            new_next_obs[:n] = self.next_obs[:n]
            new_actions[:n] = self.actions[:n]
            new_rewards[:n] = self.rewards[:n]
            new_dones[:n] = self.dones[:n]

        self.obs = new_obs
        self.next_obs = new_next_obs
        self.actions = new_actions
        self.rewards = new_rewards
        self.dones = new_dones
        self._capacity = capacity

    def _maybe_grow(self):
        """Grow backing arrays if the current capacity is exhausted."""
        if self.size < self._capacity or self._capacity >= self.max_size:
            return
        self._allocate(min(self._capacity * self._GROWTH_FACTOR, self.max_size))

    # ------------------------------------------------------------------
    # Public API  (unchanged from original)
    # ------------------------------------------------------------------

    def add(self, obs, action, reward, next_obs, done):
        """Add a new experience to the buffer.

        Args:
            obs: Observation
            action: Action
            reward: Reward
            next_obs: Next observation
            done: Done
        """
        self._maybe_grow()

        self.obs[self.ptr] = np.array(obs).copy()
        self.next_obs[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.dones[self.ptr] = np.array(done).copy()
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, replace=True, use_cer=False, to_tensor=False, device=None):
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size: Batch size
            replace: Whether to sample with replacement
            use_cer: Whether to use CER
            to_tensor: Whether to convert the data to PyTorch tensors
            device: Device to use

        Returns:
            A tuple of (observations, actions, rewards, next observations, dones)

        """
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if use_cer:
            inds[0] = self.ptr - 1  # always use last experience
        experience_tuples = (
            self.obs[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
        )
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x, device=device), experience_tuples))
        else:
            return experience_tuples

    def sample_obs(self, batch_size, replace=True, to_tensor=False, device=None):
        """Sample a batch of observations from the buffer.

        Args:
            batch_size: Batch size
            replace: Whether to sample with replacement
            to_tensor: Whether to convert the data to PyTorch tensors
            device: Device to use

        Returns:
            A batch of observations
        """
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if to_tensor:
            return th.tensor(self.obs[inds], device=device)
        else:
            return self.obs[inds]

    def get_all_data(self, max_samples=None):
        """Get all the data in the buffer (with a maximum specified).

        Args:
            max_samples: Maximum number of samples to return

        Returns:
            A tuple of (observations, actions, rewards, next observations, dones)
        """
        if max_samples is not None:
            inds = np.random.choice(self.size, min(max_samples, self.size), replace=False)
        else:
            inds = np.arange(self.size)
        return (
            self.obs[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
        )

    def __len__(self):
        """Get the size of the buffer."""
        return self.size

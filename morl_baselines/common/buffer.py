"""Replay buffer for multi-objective reinforcement learning."""

from typing import NamedTuple

import numpy as np
import torch as th


class ReplayBufferSamplesNp(NamedTuple):
    """Samples from the replay buffer in numpy format."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    idxes: np.ndarray


class ReplayBuffer:
    """Multi-objective replay buffer for multi-objective reinforcement learning."""

    def __init__(
        self,
        obs_shape,
        action_dim,
        rew_dim=1,
        max_size=100000,
        obs_dtype=np.float32,
        action_dtype=np.float32,
    ):
        """Initialize the replay buffer.

        Args:
            obs_shape: Shape of the observations
            action_dim: Dimension of the actions
            rew_dim: Dimension of the rewards
            max_size: Maximum size of the buffer
            obs_dtype: Data type of the observations
            action_dtype: Data type of the actions
        """
        self.max_size = max_size
        self.ptr, self.size = 0, 0
        self.obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.next_obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((max_size, action_dim), dtype=action_dtype)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        """Add a new experience to the buffer.

        Args:
            obs: Observation
            action: Action
            reward: Reward
            next_obs: Next observation
            done: Done
        """
        self.obs[self.ptr] = np.array(obs).copy()
        self.next_obs[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.dones[self.ptr] = np.array(done).copy()
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, obs, actions, rewards, next_obs, dones):
        """Add a batch of experiences to the buffer.

        Args:
            obs: Observations
            actions: Actions
            rewards: Rewards
            next_obs: Next observations
            dones: Dones
        """
        batch_size = len(obs)
        if self.ptr + batch_size <= self.max_size:
            self.obs[self.ptr : self.ptr + batch_size] = np.array(obs).copy()
            self.next_obs[self.ptr : self.ptr + batch_size] = np.array(next_obs).copy()
            self.actions[self.ptr : self.ptr + batch_size] = np.array(actions).copy()
            self.rewards[self.ptr : self.ptr + batch_size] = np.array(rewards).copy()
            self.dones[self.ptr : self.ptr + batch_size] = np.array(dones).reshape(-1, 1).copy()
        else:
            # Overlap case
            first_part = self.max_size - self.ptr
            second_part = batch_size - first_part
            self.obs[self.ptr :] = np.array(obs[:first_part]).copy()
            self.next_obs[self.ptr :] = np.array(next_obs[:first_part]).copy()
            self.actions[self.ptr :] = np.array(actions[:first_part]).copy()
            self.rewards[self.ptr :] = np.array(rewards[:first_part]).copy()
            self.dones[self.ptr :] = np.array(dones[:first_part]).reshape(-1, 1).copy()
 
            self.obs[:second_part] = np.array(obs[first_part:]).copy()
            self.next_obs[:second_part] = np.array(next_obs[first_part:]).copy()
            self.actions[:second_part] = np.array(actions[first_part:]).copy()
            self.rewards[:second_part] = np.array(rewards[first_part:]).copy()
            self.dones[:second_part] = np.array(dones[first_part:]).reshape(-1, 1).copy()

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size, replace=True, use_cer=False, to_tensor=False, device=None):
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size: Batch size
            replace: Whether to sample with replacement
            use_cer: Whether to use CER
            to_tensor: Whether to convert the data to PyTorch tensors
            device: Device to use

        Returns:
            A tuple of (observations, actions, rewards, next observations, dones, idxes)

        """
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if use_cer:
            inds[0] = self.ptr - 1  # always use last experience
        experience_tuples = ReplayBufferSamplesNp(
            self.obs[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
            inds,
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

class TensorReplayBuffer:
    """Multi-objective replay buffer using PyTorch tensors for GPU-native storage."""

    def __init__(
        self,
        obs_shape,
        action_dim,
        rew_dim=1,
        max_size=100000,
        device="cpu",
    ):
        """Initialize the replay buffer.
 
        Args:
            obs_shape: Shape of the observations
            action_dim: Dimension of the actions
            rew_dim: Dimension of the rewards
            max_size: Maximum size of the buffer
            device: Device to store the tensors on
        """
        self.max_size = max_size
        self.device = device
        self.ptr, self.size = 0, 0
        self.obs = th.zeros((max_size,) + obs_shape, device=device, dtype=th.float32)
        self.next_obs = th.zeros((max_size,) + obs_shape, device=device, dtype=th.float32)
        self.actions = th.zeros((max_size, action_dim), device=device, dtype=th.float32)
        self.rewards = th.zeros((max_size, rew_dim), device=device, dtype=th.float32)
        self.dones = th.zeros((max_size, 1), device=device, dtype=th.float32)
        self.weights = None # Optional weights field

    def add(self, obs, action, reward, next_obs, done, weight=None):
        """Add a new experience to the buffer."""
        self.obs[self.ptr] = th.as_tensor(obs, device=self.device, dtype=th.float32)
        self.next_obs[self.ptr] = th.as_tensor(next_obs, device=self.device, dtype=th.float32)
        self.actions[self.ptr] = th.as_tensor(action, device=self.device, dtype=th.float32)
        self.rewards[self.ptr] = th.as_tensor(reward, device=self.device, dtype=th.float32)
        self.dones[self.ptr] = th.as_tensor(done, device=self.device, dtype=th.float32)
        if weight is not None:
            if self.weights is None:
                self.weights = th.zeros((self.max_size, len(weight)), device=self.device, dtype=th.float32)
            self.weights[self.ptr] = th.as_tensor(weight, device=self.device, dtype=th.float32)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, obs, actions, rewards, next_obs, dones, weights=None):
        """Add a batch of experiences to the buffer."""
        batch_size = len(obs)
        if self.ptr + batch_size <= self.max_size:
            self.obs[self.ptr : self.ptr + batch_size] = th.as_tensor(obs, device=self.device)
            self.next_obs[self.ptr : self.ptr + batch_size] = th.as_tensor(next_obs, device=self.device)
            self.actions[self.ptr : self.ptr + batch_size] = th.as_tensor(actions, device=self.device)
            self.rewards[self.ptr : self.ptr + batch_size] = th.as_tensor(rewards, device=self.device)
            self.dones[self.ptr : self.ptr + batch_size] = th.as_tensor(dones, device=self.device).reshape(-1, 1).float()
            if weights is not None:
                if self.weights is None:
                    self.weights = th.zeros((self.max_size, weights.shape[-1]), device=self.device, dtype=th.float32)
                self.weights[self.ptr : self.ptr + batch_size] = th.as_tensor(weights, device=self.device)
        else:
            # Overlap case
            first_part = self.max_size - self.ptr
            second_part = batch_size - first_part
            self.obs[self.ptr :] = th.as_tensor(obs[:first_part], device=self.device)
            self.next_obs[self.ptr :] = th.as_tensor(next_obs[:first_part], device=self.device)
            self.actions[self.ptr :] = th.as_tensor(actions[:first_part], device=self.device)
            self.rewards[self.ptr :] = th.as_tensor(rewards[:first_part], device=self.device)
            self.dones[self.ptr :] = th.as_tensor(dones[:first_part], device=self.device).reshape(-1, 1).float()
            if weights is not None:
                if self.weights is None:
                    self.weights = th.zeros((self.max_size, weights.shape[-1]), device=self.device, dtype=th.float32)
                self.weights[self.ptr :] = th.as_tensor(weights[:first_part], device=self.device)
 
            self.obs[:second_part] = th.as_tensor(obs[first_part:], device=self.device)
            self.next_obs[:second_part] = th.as_tensor(next_obs[first_part:], device=self.device)
            self.actions[:second_part] = th.as_tensor(actions[first_part:], device=self.device)
            self.rewards[:second_part] = th.as_tensor(rewards[first_part:], device=self.device)
            self.dones[:second_part] = th.as_tensor(dones[first_part:], device=self.device).reshape(-1, 1).float()
            if weights is not None:
                self.weights[:second_part] = th.as_tensor(weights[first_part:], device=self.device)
 
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size, replace=True, to_tensor=True, device=None):
        """Sample a batch of experiences from the buffer."""
        device = device or self.device
        inds = th.randint(0, self.size, (batch_size,), device="cpu") # Indices are small
        if self.weights is not None:
            return (
                self.obs[inds].to(device),
                self.actions[inds].to(device),
                self.weights[inds].to(device),
                self.rewards[inds].to(device),
                self.next_obs[inds].to(device),
                self.dones[inds].to(device),
            )
        return (
            self.obs[inds].to(device),
            self.actions[inds].to(device),
            self.rewards[inds].to(device),
            self.next_obs[inds].to(device),
            self.dones[inds].to(device),
        )

    def sample_obs(self, batch_size, replace=True, to_tensor=True, device=None):
        """Sample a batch of observations from the buffer."""
        device = device or self.device
        inds = th.randint(0, self.size, (batch_size,), device="cpu")
        return self.obs[inds].to(device)

    def __len__(self):
        """Get the size of the buffer."""
        return self.size

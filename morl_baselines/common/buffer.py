"""Replay buffer for multi-objective reinforcement learning."""
import copy
import os

import numpy as np
import torch as th

from morl_baselines.common.observation import Observation


class ReplayBuffer:
    """Multi-objective replay buffer for multi-objective reinforcement learning."""

    def __init__(
        self,
        action_shape,
        rew_dim=1,
        max_size=100000,
        action_dtype=np.float32,
    ):
        """Initialize the replay buffer.

        Args:
            action_shape: Dimension of the actions
            rew_dim: Dimension of the rewards
            max_size: Maximum size of the buffer
            action_dtype: Data type of the actions
        """
        self.max_size = max_size
        self.ptr, self.size = 0, 0
        self.obs = np.zeros((max_size,), dtype=Observation)
        self.next_obs = np.zeros((max_size,), dtype=object)
        self.actions = np.zeros((max_size,) + action_shape, dtype=action_dtype)
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
        self.obs[self.ptr] = copy.deepcopy(obs)  # We could try to first call a .copy() method of the observation if implemented here, but it may be extra
        self.next_obs[self.ptr] = copy.deepcopy(next_obs)
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
            return (
                np.array([observation.to_tensor(device=device) for observation in experience_tuples[0]]),
                th.tensor(experience_tuples[1], device=device),
                th.tensor(experience_tuples[2], device=device),
                np.array([observation.to_tensor(device=device) for observation in experience_tuples[3]]),
                th.tensor(experience_tuples[4], device=device),
            )
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
            return np.array([observation.to_tensor(device=device) for observation in self.obs[inds]])
        else:
            return self.obs[inds]

    def get_all_data(self, max_samples=None, to_tensor=False, device=None):
        """Get all the data in the buffer (with a maximum specified).

        Args:
            max_samples: Maximum number of samples to return
            to_tensor: Whether to convert the data to PyTorch tensors
            device: Device to use

        Returns:
            A tuple of (observations, actions, rewards, next observations, dones)
        """
        if max_samples is not None:
            inds = np.random.choice(self.size, min(max_samples, self.size), replace=False)
        else:
            inds = np.arange(self.size)
        samples = (
            self.obs[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
        )

        if to_tensor:
            return (
                np.array([observation.to_tensor(device=device) for observation in samples[0]]),
                th.tensor(samples[1], device=device),
                th.tensor(samples[2], device=device),
                np.array([observation.to_tensor(device=device) for observation in samples[3]]),
                th.tensor(samples[4], device=device),
            )
        else:
            return samples

    def save(self, path):
        """Save the buffer to a file.

        Args:
            path: Path to the file
        """

        if not os.path.isdir(path):
            os.makedirs(path)

        np.savez_compressed(
            path + "buffer_without_obs.npz",
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            ptr=self.ptr,
            size=self.size,
        )
        # Save the observations
        # We save the observations separately because they can be large, as we don't know their type (maybe handle the case of np.ndarray separately?)
        if not os.path.isdir(path + "obs"):
            os.makedirs(path + "obs")
        for i, obs in enumerate(self.obs):
            obs.save(path + "obs/" + str(i))
        if not os.path.isdir(path + "next_obs"):
            os.makedirs(path + "next_obs")
        for i, obs in enumerate(self.next_obs):
            obs.save(path + "next_obs/" + str(i))

    def load(self, path):
        """Load the buffer from a file.

        Args:
            path: Path to the file
        """

        data = np.load(path, allow_pickle=True)
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.dones = data["dones"]
        self.ptr = data["ptr"]
        self.size = data["size"]

        # Load the observations
        self.obs = np.zeros((self.max_size,), dtype=Observation)
        self.next_obs = np.zeros((self.max_size,), dtype=Observation)

        for i in range(self.size):
            self.obs[i] = Observation().load(path + "obs/" + str(i))
            self.next_obs[i] = Observation().load(path + "next_obs/" + str(i))

    def __len__(self):
        """Get the size of the buffer."""
        return self.size

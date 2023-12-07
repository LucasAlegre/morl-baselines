"""Prioritized Replay Buffer.

Code adapted from https://github.com/sfujim/LAP-PAL

Code modified to fit the MoDMSE environement
"""
import copy
import os

import numpy as np
import torch as th

from morl_baselines.common.observation import Observation


class SumTree:
    """SumTree with fixed size."""

    def __init__(self, max_size):
        """Initialize the SumTree.

        Args:
            max_size: Maximum size of the SumTree
        """
        self.nodes = []
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        for _ in range(int(np.ceil(np.log2(max_size))) + 1):
            nodes = np.zeros(level_size)
            self.nodes.append(nodes)
            level_size *= 2

    def sample(self, batch_size):
        """Batch binary search through sum tree. Sample a priority between 0 and the max priority and then search the tree for the corresponding index.

        Args:
            batch_size: Number of indices to sample

        Returns:
            indices: Indices of the sampled nodes

        """
        query_value = np.random.uniform(0, self.nodes[0][0], size=batch_size)
        node_index = np.zeros(batch_size, dtype=int)

        for nodes in self.nodes[1:]:
            node_index *= 2
            left_sum = nodes[node_index]

            is_greater = np.greater(query_value, left_sum)
            # If query_value > left_sum -> go right (+1), else go left (+0)
            node_index += is_greater
            # If we go right, we only need to consider the values in the right tree
            # so we subtract the sum of values in the left tree
            query_value -= left_sum * is_greater

        return node_index

    def set(self, node_index, new_priority):
        """Set the priority of node at node_index to new_priority.

        Args:
            node_index: Index of the node to update
            new_priority: New priority of the node
        """
        priority_diff = new_priority - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2

    def batch_set(self, node_index, new_priority):
        """Batched version of set.

        Args:
            node_index: Index of the nodes to update
            new_priority: New priorities of the nodes
        """
        # Confirm we don't increment a node twice
        node_index, unique_index = np.unique(node_index, return_index=True)
        priority_diff = new_priority[unique_index] - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2


class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer."""

    def __init__(
            self,
            action_dim=1,
            rew_dim=5,
            max_size=100000,
            action_dtype=np.float32,
            min_priority=1e-5,
    ):
        """Initialize the Prioritized Replay Buffer.

        Args:
            action_dim: Dimension of the actions
            rew_dim: Dimension of the rewards
            max_size: Maximum size of the buffer
            action_dtype: Data type of the actions
            min_priority: Minimum priority of the buffer
        """
        self.max_size = max_size
        (
            self.ptr,
            self.size,
        ) = (
            0,
            0,
        )
        self.obs = np.zeros((max_size,), dtype=Observation)
        self.next_obs = np.zeros((max_size,), dtype=Observation)
        self.actions = np.zeros((max_size, action_dim), dtype=action_dtype)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

        self.tree = SumTree(max_size)
        self.min_priority = min_priority

    def add(self, obs, action, reward, next_obs, done, priority=None):
        """Add a new experience to the buffer.

        Args:
            obs: Observation
            action: Action
            reward: Reward
            next_obs: Next observation
            done: Done
            priority: Priority of the new experience

        """
        self.obs[self.ptr] = copy.deepcopy(
            obs)  # We could try to first call a .copy() method of the observation if implemented here, but it may be extra
        self.next_obs[self.ptr] = copy.deepcopy(next_obs)
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.dones[self.ptr] = np.array(done).copy()

        self.tree.set(self.ptr, self.min_priority if priority is None else priority)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, to_tensor=False, device=None):
        """Sample a batch of experience tuples from the buffer.

        Args:
            batch_size: Number of experiences to sample
            to_tensor:  Whether to convert the batch to a tensor
            device: Device to move the tensor to

        Returns:
            batch: Batch of experiences
        """
        idxes = self.tree.sample(batch_size)

        experience_tuples = (
            self.obs[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.next_obs[idxes],
            self.dones[idxes],
        )
        if to_tensor:
            return (
                np.array([observation.to_tensor(device=device) for observation in experience_tuples[0]]),
                th.tensor(experience_tuples[1]).to(device),
                th.tensor(experience_tuples[2]).to(device),
                np.array([observation.to_tensor(device=device) for observation in experience_tuples[3]]),
                th.tensor(experience_tuples[4]).to(device),
                idxes,
            )
        else:
            return experience_tuples + (idxes,)

    def sample_obs(self, batch_size, to_tensor=False, device=None):
        """Sample a batch of observations from the buffer.

        Args:
            batch_size: Number of observations to sample
            to_tensor: Whether to convert the batch to a tensor
            device: Device to move the tensor to

        Returns:
            batch: Batch of observations
        """
        idxes = self.tree.sample(batch_size)
        if to_tensor:
            return np.array([observation.to_tensor(device=device) for observation in self.obs[idxes]])
        else:
            return self.obs[idxes]

    def update_priorities(self, idxes, priorities):
        """Update the priorities of the experiences at idxes.

        Args:
            idxes: Indexes of the experiences to update
            priorities: New priorities of the experiences
        """
        self.min_priority = max(self.min_priority, priorities.max())
        self.tree.batch_set(idxes, priorities)

    def get_all_data(self, max_samples=None, to_tensor=False, device=None):
        """Get all the data in the buffer.

        Args:
            max_samples: Maximum number of samples to return
            to_tensor: Whether to convert the batch to a tensor
            device: Device to move the tensor to

        Returns:
            batch: Batch of experiences
        """
        if max_samples is not None and max_samples < self.size:
            inds = np.random.choice(self.size, max_samples, replace=False)
        else:
            inds = np.arange(self.size)
        tuples = (
            self.obs[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
        )
        if to_tensor:
            return (
                np.array([observation.to_tensor(device=device) for observation in tuples[0]]),
                th.tensor(tuples[1], device=device),
                th.tensor(tuples[2], device=device),
                np.array([observation.to_tensor(device=device) for observation in tuples[3]]),
                th.tensor(tuples[4], device=device),
            )
        else:
            return tuples

    def __len__(self):
        """Return the size of the buffer."""
        return self.size

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
            tree=self.tree.nodes,
            min_priority=self.min_priority,
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
        self.tree.nodes = data["tree"]
        self.min_priority = data["min_priority"]
        self.ptr = data["ptr"]
        self.size = data["size"]

        # Load the observations
        self.obs = np.zeros((self.max_size,), dtype=Observation)
        self.next_obs = np.zeros((self.max_size,), dtype=Observation)

        for i in range(self.size):
            self.obs[i] = Observation().load(path + "obs/" + str(i))
            self.next_obs[i] = Observation().load(path + "next_obs/" + str(i))

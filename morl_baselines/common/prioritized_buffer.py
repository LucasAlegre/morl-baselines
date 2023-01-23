import numpy as np
import torch as th


# Code adapted from https://github.com/sfujim/LAP-PAL
class SumTree:
    def __init__(self, max_size):
        self.nodes = []
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        for _ in range(int(np.ceil(np.log2(max_size))) + 1):
            nodes = np.zeros(level_size)
            self.nodes.append(nodes)
            level_size *= 2

    # Batch binary search through sum tree
    # Sample a priority between 0 and the max priority
    # and then search the tree for the corresponding index
    def sample(self, batch_size):
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
        priority_diff = new_priority - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2

    def batch_set(self, node_index, new_priority):
        # Confirm we don't increment a node twice
        node_index, unique_index = np.unique(node_index, return_index=True)
        priority_diff = new_priority[unique_index] - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2


class PrioritizedReplayBuffer:
    def __init__(
        self,
        obs_shape,
        action_dim,
        rew_dim=1,
        max_size=100000,
        obs_dtype=np.float32,
        action_dtype=np.float32,
        eps=1e-5,
    ):
        self.max_size = max_size
        self.ptr, self.size, = (
            0,
            0,
        )
        self.obs = np.zeros((max_size,) + (obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((max_size,) + (obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((max_size, action_dim), dtype=action_dtype)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

        self.tree = SumTree(max_size)
        self.eps = eps
        self.max_priority = eps

    def add(self, obs, action, reward, next_obs, done, priority=None):
        self.obs[self.ptr] = np.array(obs).copy()
        self.next_obs[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.dones[self.ptr] = np.array(done).copy()

        self.tree.set(self.ptr, self.max_priority if priority is None else priority)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, to_tensor=False, device=None):
        idxes = self.tree.sample(batch_size)

        experience_tuples = (
            self.obs[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.next_obs[idxes],
            self.dones[idxes],
        )
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples)) + (idxes,)  # , weights)
        else:
            return experience_tuples + (idxes,)

    def sample_obs(self, batch_size, to_tensor=False, device=None):
        idxes = self.tree.sample(batch_size)
        if to_tensor:
            return th.tensor(self.obs[idxes]).to(device)
        else:
            return self.obs[idxes]

    def update_priorities(self, idxes, priorities):
        self.max_priority = max(self.max_priority, priorities.max())
        self.tree.batch_set(idxes, priorities)

    def get_all_data(self, max_samples=None, to_tensor=False, device=None):
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
            return tuple(map(lambda x: th.tensor(x).to(device), tuples))
        else:
            return tuples

    def __len__(self):
        return self.size

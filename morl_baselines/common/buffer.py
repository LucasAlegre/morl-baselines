from typing import Union

import numpy as np
import torch as th


class ReplayBuffer:
    def __init__(
            self,
            obs_shape,
            action_dim,
            rew_dim=1,
            max_size=100000,
            obs_dtype=np.float32,
            action_dtype=np.float32,
    ):
        self.max_size = max_size
        self.ptr, self.size = 0, 0
        self.obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.next_obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((max_size, action_dim), dtype=action_dtype)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = np.array(obs).copy()
        self.next_obs[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.dones[self.ptr] = np.array(done).copy()
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, replace=True, use_cer=False, to_tensor=False, device=None):
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
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples))
        else:
            return experience_tuples

    def sample_obs(self, batch_size, replace=True, to_tensor=False, device=None):
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if to_tensor:
            return th.tensor(self.obs[inds]).to(device)
        else:
            return self.obs[inds]

    def get_all_data(self, max_samples=None):
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
        return self.size


class PPOReplayBuffer:
    def __init__(self, size: int, num_envs: int, obs_shape: tuple, action_shape: tuple, reward_dim: int, device: Union[th.device, str] = "auto"):
        self.size = size
        self.ptr = 0
        self.num_envs = num_envs
        self.obs = th.zeros((self.size, self.num_envs) + obs_shape).to(device)
        self.actions = th.zeros((self.size, self.num_envs) + action_shape).to(device)
        self.logprobs = th.zeros((self.size, self.num_envs)).to(device)
        self.rewards = th.zeros((self.size, self.num_envs, reward_dim), dtype=th.float32).to(device)
        self.dones = th.zeros((self.size, self.num_envs)).to(device)
        self.values = th.zeros((self.size, self.num_envs, reward_dim), dtype=th.float32).to(device)

    def add(self, obs, actions, logprobs, rewards, dones, values):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.logprobs[self.ptr] = logprobs
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.ptr = (self.ptr + 1) % self.size

    def get(self, step):
        return (
            self.obs[step],
            self.actions[step],
            self.logprobs[step],
            self.rewards[step],
            self.dones[step],
            self.values[step]
        )

    def get_all(self):
        return (
            self.obs,
            self.actions,
            self.logprobs,
            self.rewards,
            self.dones,
            self.values
        )

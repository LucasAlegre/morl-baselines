import numpy as np
import torch as th


class AccruedRewardReplayBuffer:
    """Replay buffer with accrued rewards stored (for ESR algorithms)"""

    def __init__(
        self,
        obs_shape,
        action_shape,
        rew_dim=1,
        max_size=100000,
        obs_dtype=np.float32,
        action_dtype=np.float32,
    ):
        self.max_size = max_size
        self.ptr, self.size = 0, 0
        self.obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.next_obs = np.zeros((max_size,) + obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((max_size,) + action_shape, dtype=action_dtype)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.accrued_rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, obs, accrued_reward, action, reward, next_obs, done):
        self.obs[self.ptr] = np.array(obs).copy()
        self.next_obs[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.accrued_rewards[self.ptr] = np.array(accrued_reward).copy()
        self.dones[self.ptr] = np.array(done).copy()
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(
        self, batch_size, replace=True, use_cer=False, to_tensor=False, device=None
    ):
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if use_cer:
            inds[0] = self.ptr - 1  # always use last experience
        experience_tuples = (
            self.obs[inds],
            self.accrued_rewards[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
        )
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples))
        else:
            return experience_tuples

    def cleanup(self):
        self.size, self.ptr = 0, 0

    def get_all_data(self, max_samples=None, to_tensor=False, device=None):
        """
        Returns the whole buffer
        :param max_samples: the number of samples to return, if not specified, returns the full buffer (ordered!)
        """
        if max_samples is not None:
            inds = np.random.choice(
                self.size, min(max_samples, self.size), replace=False
            )
        else:
            inds = np.arange(self.size)
        experience_tuples = (
            self.obs[inds],
            self.accrued_rewards[inds],
            self.actions[inds],
            self.rewards[inds],
            self.next_obs[inds],
            self.dones[inds],
        )
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples))
        else:
            return experience_tuples

    def __len__(self):
        return self.size

import mo_gymnasium as mo_gym
import numpy as np
import torch as th
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.common.evaluation import eval_mo_reward_conditioned
from morl_baselines.single_policy.esr.eupg import EUPG


if __name__ == "__main__":
    env = MORecordEpisodeStatistics(mo_gym.make("fishwood-v0"), gamma=0.99)
    eval_env = mo_gym.make("fishwood-v0")

    def scalarization(reward: np.ndarray, w=None):
        reward = th.tensor(reward) if not isinstance(reward, th.Tensor) else reward
        # Handle the case when reward is a single tensor of shape (2, )
        if reward.dim() == 1 and reward.size(0) == 2:
            return min(reward[0], reward[1] // 2).item()

        # Handle the case when reward is a tensor of shape (200, 2)
        elif reward.dim() == 2 and reward.size(1) == 2:
            return th.min(reward[:, 0], reward[:, 1] // 2)

    agent = EUPG(env, scalarization=scalarization, weights=np.ones(2), gamma=0.99, log=True, learning_rate=0.001)
    agent.train(total_timesteps=int(4e6), eval_env=eval_env, eval_freq=1000)

    print(eval_mo_reward_conditioned(agent, env=eval_env, scalarization=scalarization))

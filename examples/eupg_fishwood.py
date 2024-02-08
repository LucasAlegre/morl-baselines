import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.utils import MORecordEpisodeStatistics

from morl_baselines.common.evaluation import eval_mo_reward_conditioned
from morl_baselines.single_policy.esr.eupg import EUPG


if __name__ == "__main__":
    env = MORecordEpisodeStatistics(mo_gym.make("fishwood-v0"), gamma=0.99)
    eval_env = mo_gym.make("fishwood-v0")

    def scalarization(reward: np.ndarray):
        return min(reward[0], reward[1] // 2)

    agent = EUPG(env, scalarization=scalarization, gamma=0.99, log=True, learning_rate=0.001)
    agent.train(total_timesteps=int(4e6), eval_env=eval_env, eval_freq=1000)

    print(eval_mo_reward_conditioned(agent, env=eval_env, scalarization=scalarization))

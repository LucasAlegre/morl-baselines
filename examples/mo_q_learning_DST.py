import time

import mo_gym
import numpy as np
from mo_gym import MORecordEpisodeStatistics

from mo_gym.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure, CONCAVE_MAP
from morl_baselines.common.scalarization import tchebicheff, weighted_sum
from morl_baselines.mo_algorithms.mo_q_learning import MOQLearning

if __name__ == "__main__":
    env = MORecordEpisodeStatistics(DeepSeaTreasure(dst_map=CONCAVE_MAP), gamma=0.9)
    eval_env = DeepSeaTreasure(dst_map=CONCAVE_MAP)
    scalarization = tchebicheff(tau=4., reward_dim=2)
    weights = np.array([0.3, 0.7])

    agent = MOQLearning(env, id=0, scalarization=weighted_sum, weights=weights)
    agent.learn(total_timesteps=int(1e5), start_time=time.time(), eval_freq=100, eval_env=eval_env)

    print(mo_gym.eval_mo(agent, env=eval_env, w=weights))
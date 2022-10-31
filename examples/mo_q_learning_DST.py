import time

import gym
import mo_gym
import numpy as np
from mo_gym.utils import MORecordEpisodeStatistics

from mo_gym.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure, CONCAVE_MAP
from morl_baselines.common.scalarization import tchebicheff, weighted_sum
from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning

if __name__ == "__main__":
    env = MORecordEpisodeStatistics(DeepSeaTreasure(dst_map=CONCAVE_MAP), gamma=0.9)
    eval_env = gym.wrappers.TimeLimit(DeepSeaTreasure(dst_map=CONCAVE_MAP), 500)
    scalarization = tchebicheff(tau=4.0, reward_dim=2)
    weights = np.array([0.3, 0.7])

    agent = MOQLearning(env, scalarization=scalarization, weights=weights, log=True)
    agent.train(
        total_timesteps=int(1e5),
        start_time=time.time(),
        eval_freq=100,
        eval_env=eval_env,
    )

    print(mo_gym.eval_mo(agent, env=eval_env, w=weights))

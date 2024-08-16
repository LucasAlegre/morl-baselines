import time

import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

from morl_baselines.common.evaluation import eval_mo
from morl_baselines.common.scalarization import tchebicheff
from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning


if __name__ == "__main__":
    env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-concave-v0"), gamma=0.99)
    eval_env = mo_gym.make("deep-sea-treasure-concave-v0")
    scalarization = tchebicheff(tau=4.0, reward_dim=2)
    weights = np.array([0.3, 0.7])

    agent = MOQLearning(env, scalarization=scalarization, weights=weights, log=True)
    agent.train(
        total_timesteps=int(1e5),
        start_time=time.time(),
        eval_freq=100,
        eval_env=eval_env,
    )

    print(eval_mo(agent, env=eval_env, w=weights))

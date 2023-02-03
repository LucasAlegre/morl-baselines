import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from mo_gymnasium import MORecordEpisodeStatistics
from mo_gymnasium.envs.deep_sea_treasure.deep_sea_treasure import (
    DEFAULT_MAP,
    DeepSeaTreasure,
)

from morl_baselines.common.scalarization import tchebicheff
from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import (
    MPMOQLearning,
)


if __name__ == "__main__":
    env = MORecordEpisodeStatistics(DeepSeaTreasure(dst_map=DEFAULT_MAP), gamma=0.9)
    eval_env = gym.wrappers.TimeLimit(DeepSeaTreasure(dst_map=DEFAULT_MAP), 500)
    scalarization = tchebicheff(tau=4.0, reward_dim=2)

    agent = MPMOQLearning(
        env,
        ref_point=np.array([0.0, -25.0]),
        scalarization=scalarization,
        num_timesteps=int(1e5),
        weights_step_size=0.1,
        initial_epsilon=0.9,
        epsilon_decay_steps=int(5e4),
    )
    agent.train()

    front, discounted_front = agent.eval_all_agents()
    print(discounted_front)
    plt.scatter(np.array(discounted_front)[:, 0], np.array(discounted_front)[:, 1])
    plt.show()

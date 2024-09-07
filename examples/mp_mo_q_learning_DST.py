import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

from morl_baselines.common.scalarization import tchebicheff
from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import (
    MPMOQLearning,
)


if __name__ == "__main__":
    env = MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-concave-v0"), gamma=0.99)
    eval_env = mo_gym.make("deep-sea-treasure-concave-v0")
    scalarization = tchebicheff(tau=4.0, reward_dim=2)

    mp_moql = MPMOQLearning(
        env,
        learning_rate=0.3,
        scalarization=scalarization,
        use_gpi_policy=False,
        dyna=False,
        initial_epsilon=1,
        final_epsilon=0.01,
        epsilon_decay_steps=int(2e5),
        weight_selection_algo="random",
        epsilon_ols=0.0,
    )
    mp_moql.train(
        total_timesteps=15 * int(2e5),
        timesteps_per_iteration=int(2e5),
        eval_freq=100,
        num_episodes_eval=1,
        eval_env=eval_env,
        ref_point=np.array([0.0, -25.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=0.99),
    )

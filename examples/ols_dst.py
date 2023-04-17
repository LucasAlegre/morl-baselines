import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import (
    MPMOQLearning,
)


def main():
    GAMMA = 0.99
    env = mo_gym.MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-v0"), gamma=GAMMA)

    eval_env = mo_gym.make("deep-sea-treasure-v0")

    mp_moql = MPMOQLearning(
        env,
        learning_rate=0.3,
        gamma=GAMMA,
        use_gpi_policy=True,
        dyna=True,
        dyna_updates=5,
        initial_epsilon=1,
        final_epsilon=0.01,
        epsilon_decay_steps=int(2e5),
        weight_selection_algo="ols",
        epsilon_ols=0.0,
    )
    mp_moql.train(
        total_timesteps=15 * int(2e5),
        eval_env=eval_env,
        ref_point=np.array([0.0, -25.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=GAMMA),
        timesteps_per_iteration=int(2e5),
        eval_freq=100,
        num_episodes_eval=1,
    )


if __name__ == "__main__":
    main()

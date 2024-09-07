import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

from morl_baselines.multi_policy.pcn.pcn import PCN


def main():
    def make_env():
        env = mo_gym.make("minecart-deterministic-v0")
        env = MORecordEpisodeStatistics(env, gamma=1.0)
        return env

    env = make_env()

    agent = PCN(
        env,
        scaling_factor=np.array([1, 1, 0.1, 0.1]),
        learning_rate=1e-3,
        batch_size=256,
        project_name="MORL-Baselines",
        experiment_name="PCN",
        log=True,
    )

    agent.train(
        eval_env=make_env(),
        total_timesteps=int(1e7),
        ref_point=np.array([0, 0, -200.0]),
        num_er_episodes=20,
        max_buffer_size=50,
        num_model_updates=50,
        max_return=np.array([1.5, 1.5, -0.0]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=1.0),
    )


if __name__ == "__main__":
    main()

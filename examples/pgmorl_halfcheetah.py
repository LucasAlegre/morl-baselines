import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
from morl_baselines.single_policy.ser.mo_ppo import make_env


if __name__ == "__main__":
    env_id = "mo-halfcheetah-v4"
    algo = PGMORL(
        env_id=env_id,
        num_envs=4,
        pop_size=6,
        warmup_iterations=80,
        evolutionary_iterations=20,
        num_weight_candidates=7,
        limit_env_steps=int(5e6),
    )
    algo.train()
    env = make_env(env_id, 422, 1, "PGMORL_test", gamma=0.995)()  # idx != 0 to avoid taking videos

    # Execution of trained policies
    for a in algo.archive.individuals:
        scalarized, discounted_scalarized, reward, discounted_reward = mo_gym.eval_mo(
            agent=a, env=env, w=np.array([1.0, 1.0]), render=True
        )
        print(f"Agent #{a.id}")
        print(f"Scalarized: {scalarized}")
        print(f"Discounted scalarized: {discounted_scalarized}")
        print(f"Vectorial: {reward}")
        print(f"Discounted vectorial: {discounted_reward}")

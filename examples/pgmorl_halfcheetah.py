import mo_gym
from morl_baselines.pgmorl.pgmorl import PGMORL

if __name__ == "__main__":
    algo = PGMORL(env_id="mo-halfcheetah-v4", num_envs=4, pop_size=3, warmup_iterations=10, evolutionary_iterations=10, limit_env_steps=1000000)
    algo.train()

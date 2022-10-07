import gym
import mo_gym
from mo_gym.utils import MORecordEpisodeStatistics
from mo_gym.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure, CONCAVE_MAP

from morl_baselines.mo_algorithms.esr.eupg import EUPG

if __name__ == "__main__":
    env = MORecordEpisodeStatistics(mo_gym.make('fishwood-v0'), gamma=0.99)
    eval_env = mo_gym.make('fishwood-v0')
    scalarization = lambda r: min(r[0], r[1] // 2)

    agent = EUPG(env, scalarization=scalarization, gamma=0.99, log=True, buffer_size=200)
    agent.train(total_timesteps=int(1e6), eval_env=eval_env, eval_freq=1000)

    print(mo_gym.eval_mo_esr(agent, env=eval_env, scalarization=scalarization))

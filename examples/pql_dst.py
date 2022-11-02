import numpy as np
from pymoo.indicators.hv import HV

import mo_gym
from mo_gym.deep_sea_treasure.deep_sea_treasure import CONCAVE_MAP
from morl_baselines.multi_policy.pareto_q_learning.pql import ParetoQ

if __name__ == "__main__":
    env_id = "deep-sea-treasure-v0"
    env = mo_gym.make(env_id, dst_map=CONCAVE_MAP)
    ref_point = np.array([0, -25])
    hypervolume = HV(ref_point=-1 * ref_point)
    perf_indic = lambda vec_set: hypervolume(-1 * np.array(list(vec_set)))  # Pymoo flips everything.

    pf = {
        (3.0, -5.0),
        (16.0, -9.0),
        (2.0, -3.0),
        (1.0, -1.0),
        (74.0, -17.0),
        (24.0, -13.0),
        (50.0, -14.0),
        (5.0, -7.0),
        (8.0, -8.0),
        (124.0, -19.0),
    }
    value = perf_indic(pf)

    agent = ParetoQ(
        env,
        perf_indic,
        gamma=1,
        init_epsilon=1,
        epsilon_decay=0.997,
        decay_every=10,
        min_epsilon=0.05,
        decimals=2,
        novec=30,
    )
    agent.train(
        iterations=3000,
        max_timesteps=100,
        log=True,
        log_every=1,
        project_name="PQL",
        experiment_name="PQL",
    )

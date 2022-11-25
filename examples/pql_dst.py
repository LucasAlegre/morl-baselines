import mo_gym
import numpy as np
from mo_gym.deep_sea_treasure.deep_sea_treasure import CONCAVE_MAP
from pymoo.indicators.hv import HV

from morl_baselines.multi_policy.pareto_q_learning.pql import PQL

if __name__ == "__main__":
    env_id = "deep-sea-treasure-v0"
    env = mo_gym.make(env_id, dst_map=CONCAVE_MAP)
    ref_point = np.array([0, -25])

    agent = PQL(
        env,
        ref_point,
        gamma=1.,
        initial_epsilon=1.,
        epsilon_decay=0.999,
        final_epsilon=0.2,
        seed=1,
        project_name="MORL-baselines",
        experiment_name="Pareto Q-Learning",
        log=False,
    )

    num_episodes = 10000
    pf = agent.train(num_episodes=10000, max_timesteps=1000, log_every=100, action_eval='pareto_cardinality')
    print(pf)

    # Execute a policy
    target = np.array(pf.pop())
    print(f'Tracking {target}')
    reward = agent.track_policy(target)
    print(f'Obtained {reward}')

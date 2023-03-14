import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy.pareto_q_learning.pql import PQL


if __name__ == "__main__":
    env = mo_gym.make("deep-sea-treasure-concave-v0")
    ref_point = np.array([0, -25])

    agent = PQL(
        env,
        ref_point,
        gamma=0.99,
        initial_epsilon=1.0,
        epsilon_decay=0.997,
        final_epsilon=0.2,
        seed=1,
        project_name="MORL-Baselines",
        experiment_name="Pareto Q-Learning",
        log=True,
    )

    num_episodes = 10000
    pf = agent.train(
        num_episodes=10000,
        log_every=100,
        action_eval="hypervolume",
        known_pareto_front=env.pareto_front(gamma=0.99),
        eval_ref_point=ref_point,
    )
    print(pf)

    # Execute a policy
    target = np.array(pf.pop())
    print(f"Tracking {target}")
    reward = agent.track_policy(target)
    print(f"Obtained {reward}")

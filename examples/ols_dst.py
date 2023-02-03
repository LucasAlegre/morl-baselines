import mo_gymnasium as mo_gym

from morl_baselines.multi_policy.ols.ols import OLS
from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning


def main():

    GAMMA = 0.99
    env = mo_gym.MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-v0"), gamma=GAMMA)

    ols = OLS(num_objectives=2, epsilon=0.0, verbose=True)
    policies = []
    while not ols.ended():
        w = ols.next_weight()

        new_policy = MOQLearning(
            env,
            weights=w,
            learning_rate=0.3,
            gamma=GAMMA,
            initial_epsilon=1,
            final_epsilon=0.01,
            epsilon_decay_steps=int(1e5),
        )
        new_policy.train(0, total_timesteps=int(2e5))

        _, _, vec, discounted_vec = new_policy.policy_eval(eval_env=env, weights=w, writer=new_policy.writer)
        policies.append(new_policy)

        removed_inds = ols.add_solution(discounted_vec, w)

        for ind in removed_inds:
            policies.pop(ind)  # remove policies that are no longer needed


if __name__ == "__main__":
    main()

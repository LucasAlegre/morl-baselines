import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.common.utils import log_all_multi_policy_metrics
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport
from morl_baselines.single_policy.ser.mo_q_learning import MOQLearning


def main():

    GAMMA = 0.99
    env = mo_gym.MORecordEpisodeStatistics(mo_gym.make("deep-sea-treasure-v0"), gamma=GAMMA)

    ols = LinearSupport(num_objectives=2, epsilon=0.0, verbose=True)
    policies = []
    global_step = 0
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
        global_step += int(2e5)

        _, _, vec, discounted_vec = new_policy.policy_eval(eval_env=env, weights=w, writer=new_policy.writer)
        policies.append(new_policy)

        removed_inds = ols.add_solution(discounted_vec, w)

        for ind in removed_inds:
            policies.pop(ind)  # remove policies that are no longer needed

        # TODO this is really annoying to do from the outside
        log_all_multi_policy_metrics(
            current_front=ols.ccs,
            hv_ref_point=np.array([0, -25]),
            reward_dim=env.unwrapped.reward_dim,
            global_step=global_step,
            writer=new_policy.writer,
            ref_front=env.unwrapped.pareto_front,
        )


if __name__ == "__main__":
    main()

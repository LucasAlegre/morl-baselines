import mo_gym
import numpy as np
from matplotlib import pyplot as plt

from mo_gym.deep_sea_treasure.deep_sea_treasure import DeepSeaTreasure, CONCAVE_MAP, DEFAULT_MAP
from morl_baselines.common.scalarization import tchebicheff, weighted_sum
from morl_baselines.multi_policy_moqlearning.mp_mo_q_learning import MPMOQLearning

if __name__ == "__main__":
    env = DeepSeaTreasure(dst_map=DEFAULT_MAP)
    eval_env = DeepSeaTreasure(dst_map=DEFAULT_MAP)
    scalarization = tchebicheff(tau=4., reward_dim=2)

    agent = MPMOQLearning(env,
                          ref_point=np.array([0., -25.]),
                          scalarization=weighted_sum,
                          num_timesteps=int(1e5),
                          weights_step_size=.1,
                          initial_epsilon=0.9,
                          epsilon_decay_steps=int(5e4)
                          )
    agent.train()

    # TODO we need to think about making the API more consistent, some agents contain multiple policies without an explicit weight
    # Update, eval eval_mo are not really consistent with those IMO
    front, discounted_front = agent.eval_all_agents()
    print(discounted_front)
    plt.scatter(np.array(discounted_front[:, 0], discounted_front[:, 1]))

#%%
from gymnasium.wrappers import TransformReward, TransformObservation, RecordVideo
import mo_gymnasium as mo_gym
import numpy as np
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPIPDContinuousAction
from morl_baselines.common.evaluation import policy_evaluation_mo, eval_mo
from morl_baselines.common.utils import equally_spaced_weights


def make_env(record_video=False, num_objectives=2):
    env = mo_gym.make('water-reservoir-v0',
                      normalized_action=True, 
                      nO=num_objectives, 
                      penalize=False, 
                      initial_state=np.array([50.0], dtype=np.float32),
                      render_mode='rgb_array' if record_video else None)
    # Scale rewards and state to avoid too big q-values and numerical issues
    env = TransformReward(env, lambda r: r * np.array([1, 0.1, 1, 0.1][:num_objectives], dtype=np.float32))
    env = TransformObservation(env, lambda obs: obs * 0.01)
    if record_video:
        env = RecordVideo(env, "videos/dam", episode_trigger=lambda x: x % 100 == 0)
    return env

env = make_env()
eval_env = make_env(record_video=False)

# When using the model based version (dyna=True, per=True), NotImplementedError is raised by:
#   morl-baselines/common/model-based/utils.py (env_id==water-reservoir-v0 is not in the if-statement in the contructor of ModelEnv)
agent = GPIPDContinuousAction(
    env=env,
    per=False,
    dyna=False,
    net_arch=[64, 64],
    gradient_updates=10,
    log=True,
)

#%%
agent.train(
    eval_env=eval_env,
    ref_point=-10*np.ones(env.reward_dim),
    total_timesteps=50000,
    timesteps_per_iter=5000,
    num_eval_episodes_for_front=50,
    num_eval_weights_for_front=32
)

#%%
""" agent.load("../weights/GPI-PD gpi-ls iter=15.tar")
gpi_returns_test_tasks = [
                    policy_evaluation_mo(agent, eval_env, w, rep=100)[3] for w in equally_spaced_weights(env.reward_dim, 32)
                ]

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.title("GPI-LS on Water Reservoir")
plt.xlabel("Objective 1")
plt.ylabel("Objective 2")
plt.scatter(*np.array(gpi_returns_test_tasks).T, label="GPI-LS")
plt.legend()
plt.show() """

# %%

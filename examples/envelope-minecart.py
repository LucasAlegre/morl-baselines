import numpy as np
import gym
from gym.wrappers.record_video import RecordVideo
import mo_gym

from morl_baselines.envelope.envelope import Envelope


def main():

    def make_env():
        env = mo_gym.make("minecart-v0")
        #env = mo_gym.LinearReward(env)
        return env

    env = make_env()
    eval_env = RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    agent = Envelope(
        env,
        max_grad_norm=0.1,
        learning_rate=3e-4,
        gamma=0.98,
        batch_size=128,
        net_arch=[256, 256, 256, 256, 256],
        buffer_size=int(2e6),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=50000,
        learning_starts=100,
        min_priority=0.01,
        envelope=True,
        gradient_updates=5,
        target_net_update_freq=1000, #1000,  # 500 reduce by gradient updates
        tau=1,
        log=True,
        project_name="MineCart",
        experiment_name="Envelope",
    )

    w = np.array([0.9, 0.0, 0.1])
    agent.learn(
            total_timesteps=100000,
            total_episodes=None,
            w=w,
            M=ols.get_ccs_weights() + ols.get_corner_weights() + [w],
            eval_env=eval_env,
            eval_freq=1000,
            reset_num_timesteps=False,
            reset_learning_starts=False
        )

if __name__ == '__main__':
    main()
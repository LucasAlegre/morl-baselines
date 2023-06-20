"""General utils for the MORL baselines."""
from typing import Any, List, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType


def linearly_decaying_value(initial_value, decay_period, step, warmup_steps, final_value):
    """Returns the current value for a linearly decaying parameter.

    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.

    Args:
        decay_period: float, the period over which the value is decayed.
        step: int, the number of training steps completed so far.
        warmup_steps: int, the number of steps taken before the value is decayed.
        final value: float, the final value to which to decay the value parameter.

    Returns:
        A float, the current value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (initial_value - final_value) * steps_left / decay_period
    value = final_value + bonus
    value = np.clip(value, min(initial_value, final_value), max(initial_value, final_value))
    return value


def unique_tol(a: List[np.ndarray], tol=1e-4) -> List[np.ndarray]:
    """Returns unique elements of a list of np.arrays, within a tolerance."""
    if len(a) == 0:
        return a
    delete = np.array([False] * len(a))
    a = np.array(a)
    for i in range(len(a)):
        if delete[i]:
            continue
        for j in range(i + 1, len(a)):
            if np.allclose(a[i], a[j], tol):
                delete[j] = True
    return list(a[~delete])


def make_gif(env, agent, weight: np.ndarray, fullpath: str, fps: int = 50, length: int = 300):
    """Render an episode and save it as a gif."""
    assert "rgb_array" in env.metadata["render_modes"], "Environment does not have rgb_array rendering."

    frames = []
    state, info = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated) and len(frames) < length:
        frame = env.render()
        frames.append(frame)
        action = agent.eval(state, weight)
        state, reward, terminated, truncated, info = env.step(action)
    env.close()

    from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(list(frames), fps=fps)
    clip.write_gif(fullpath + ".gif", fps=fps)
    print("Saved gif at: " + fullpath + ".gif")


class MaxAndSkipObservationV0(gym.Wrapper):
    """This wrapper will return only every ``skip``-th frame (frameskipping) and return the max between the two last observations.

    Note: This wrapper is based on the wrapper from stable-baselines3: https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html#MaxAndSkipEnv
    """

    def __init__(self, env: gym.Env[ObsType, ActType], skip: int = 4):
        """This wrapper will return only every ``skip``-th frame (frameskipping) and return the max between the two last frames.

        Args:
            env (Env): The environment to apply the wrapper
            skip: The number of frames to skip
        """
        gym.Wrapper.__init__(self, env)

        if not np.issubdtype(type(skip), np.integer):
            raise TypeError(f"The skip is expected to be an integer, actual type: {type(skip)}")
        if skip < 2:
            raise ValueError(f"The skip value needs to be equal or greater than two, actual value: {skip}")
        if env.observation_space.shape is None:
            raise ValueError("The observation space must have the shape attribute.")

        self._skip = skip
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the environment with the given action for ``skip`` steps.

        Repeat action, sum reward, and max over last observations.

        Args:
            action: The action to step through the environment with
        Returns:
            Max of the last two observations, reward, terminated, truncated, and info from the environment
        """
        total_reward = np.zeros(self.env.reward_dim, dtype=np.float32)
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

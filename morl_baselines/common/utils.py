"""General utils for the MORL baselines."""

import math
import os
from typing import Callable, List

import numpy as np


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


def nearest_neighbors(
    n: int,
    current_weight: np.ndarray,
    all_weights: List[np.ndarray],
    dist_metric: Callable[[np.ndarray, np.ndarray], float],
) -> List[int]:
    """Returns the n closest neighbors of current_weight in all_weights, according to similarity metric.

    Args:
        n: number of neighbors
        current_weight: weight vector where we want the nearest neighbors
        all_weights: all the possible weights, can contain current_weight as well
        dist_metric: distance metric
    Return:
        the ids of the nearest neighbors in all_weights
    """
    assert n < len(all_weights)
    current_weight_tuple = tuple(current_weight)
    nearest_neighbors_ids = []
    nearest_neighbors = []

    while len(nearest_neighbors_ids) < n:
        closest_neighb_id = -1
        closest_neighb = np.zeros_like(current_weight)
        closest_neigh_dist = math.inf

        for i, w in enumerate(all_weights):
            w_tuple = tuple(w)
            if w_tuple not in nearest_neighbors and current_weight_tuple != w_tuple:
                if closest_neigh_dist > dist_metric(current_weight, w):
                    closest_neighb = w
                    closest_neighb_id = i
                    closest_neigh_dist = dist_metric(current_weight, w)
        nearest_neighbors.append(tuple(closest_neighb))
        nearest_neighbors_ids.append(closest_neighb_id)

    return nearest_neighbors_ids


def reset_wandb_env():
    """Reset the wandb environment variables.

    This is useful when running multiple sweeps in parallel, as wandb
    will otherwise try to use the same directory for all the runs.
    """
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

"""Utilities related to evaluation."""

from typing import Optional, Tuple

import numpy as np


def eval_mo(
    agent,
    env,
    w: Optional[np.ndarray] = None,
    scalarization=np.dot,
    render: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates one episode of the agent in the environment.

    Args:
        agent: Agent
        env: MO-Gymnasium environment with LinearReward wrapper
        scalarization: scalarization function, taking weights and reward as parameters
        w (np.ndarray): Weight vector
        render (bool, optional): Whether to render the environment. Defaults to False.

    Returns:
        (float, float, np.ndarray, np.ndarray): Scalarized return, scalarized discounted return, vectorized return, vectorized discounted return
    """
    obs, _ = env.reset()
    done = False
    vec_return, disc_vec_return = np.zeros_like(w), np.zeros_like(w)
    gamma = 1.0
    while not done:
        if render:
            env.render()
        obs, r, terminated, truncated, info = env.step(agent.eval(obs, w))
        done = terminated or truncated
        vec_return += r
        disc_vec_return += gamma * r
        gamma *= agent.gamma

    if w is None:
        scalarized_return = scalarization(vec_return)
        scalarized_discounted_return = scalarization(disc_vec_return)
    else:
        scalarized_return = scalarization(w, vec_return)
        scalarized_discounted_return = scalarization(w, disc_vec_return)

    return (
        scalarized_return,
        scalarized_discounted_return,
        vec_return,
        disc_vec_return,
    )


def eval_mo_reward_conditioned(
    agent,
    env,
    scalarization=np.dot,
    w: Optional[np.ndarray] = None,
    render: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates one episode of the agent in the environment. This makes the assumption that the agent is conditioned on the accrued reward i.e. for ESR agent.

    Args:
        agent: Agent
        env: MO-Gymnasium environment
        scalarization: scalarization function, taking weights and reward as parameters
        w: weight vector
        render (bool, optional): Whether to render the environment. Defaults to False.

    Returns:
        (float, float, np.ndarray, np.ndarray): Scalarized return, scalarized discounted return, vectorized return, vectorized discounted return
    """
    obs, _ = env.reset()
    done = False
    vec_return, disc_vec_return = np.zeros(env.reward_space.shape[0]), np.zeros(env.reward_space.shape[0])
    gamma = 1.0
    while not done:
        if render:
            env.render()
        obs, r, terminated, truncated, info = env.step(agent.eval(obs, disc_vec_return))
        done = terminated or truncated
        vec_return += r
        disc_vec_return += gamma * r
        gamma *= agent.gamma
    if w is None:
        scalarized_return = scalarization(vec_return)
        scalarized_discounted_return = scalarization(disc_vec_return)
    else:
        scalarized_return = scalarization(w, vec_return)
        scalarized_discounted_return = scalarization(w, disc_vec_return)

    return (
        scalarized_return,
        scalarized_discounted_return,
        vec_return,
        disc_vec_return,
    )


def policy_evaluation_mo(agent, env, w: np.ndarray, rep: int = 5) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates the value of a policy by running the policy for multiple episodes. Returns the average returns.

    Args:
        agent: Agent
        env: MO-Gymnasium environment
        w (np.ndarray): Weight vector
        rep (int, optional): Number of episodes for averaging. Defaults to 5.

    Returns:
        (float, float, np.ndarray, np.ndarray): Avg scalarized return, Avg scalarized discounted return, Avg vectorized return, Avg vectorized discounted return
    """
    evals = [eval_mo(agent, env, w) for _ in range(rep)]
    avg_scalarized_return = np.mean([eval[0] for eval in evals])
    avg_scalarized_discounted_return = np.mean([eval[1] for eval in evals])
    avg_vec_return = np.mean([eval[2] for eval in evals], axis=0)
    avg_disc_vec_return = np.mean([eval[3] for eval in evals], axis=0)

    return (
        avg_scalarized_return,
        avg_scalarized_discounted_return,
        avg_vec_return,
        avg_disc_vec_return,
    )

"""Scalarization functions relying on numpy."""
import numpy as np
from pymoo.decomposition.tchebicheff import Tchebicheff


def weighted_sum(reward: np.ndarray, weights: np.ndarray) -> float:
    """Weighted sum scalarization (numpy dot product).

    Args:
        reward: Reward vector
        weights: Weight vector

    Returns:
        float: Weighted sum
    """
    return np.dot(reward, weights)


def tchebicheff(tau: float, reward_dim: int):
    """Tchebicheff scalarization function.

    This function requires a reference point. It is automatically adapted to the best value seen so far for each component of the reward.

    Args:
        tau: Parameter to be sure the reference point is always dominating (automatically adapted).
        reward_dim: Dimension of the reward vector

    Returns:
        Callable: Tchebicheff scalarization function
    """
    best_so_far = [float("-inf") for _ in range(reward_dim)]
    tch = Tchebicheff()

    def thunk(reward: np.ndarray, weights: np.ndarray):
        for i, r in enumerate(reward):
            if best_so_far[i] < r + tau:
                best_so_far[i] = r + tau
        return -tch.do(F=reward, weights=weights, utopian_point=np.array(best_so_far))[0][0]

    return thunk

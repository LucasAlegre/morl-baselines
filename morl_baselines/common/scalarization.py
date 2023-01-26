"""Scalarization functions relying on numpy."""
import numpy as np
from pymoo.decomposition.tchebicheff import Tchebicheff


def weighted_sum(weights: np.ndarray, reward: np.ndarray) -> float:
    """Weighted sum scalarization (numpy dot product).

    Args:
        weights: Weight vector
        reward: Reward vector

    Returns:
        float: Weighted sum
    """
    return np.dot(weights, reward)


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

    def thunk(weights: np.ndarray, reward: np.ndarray):
        for i, r in enumerate(reward):
            if best_so_far[i] < r + tau:
                best_so_far[i] = r + tau
        # Losses have to be positive for gradient descent in EUPG, hence 1/distance instead of -distance
        return 1 / tch.do(F=reward, weights=weights, utopian_point=np.array(best_so_far))[0][0]

    return thunk

import numpy as np
from pymoo.decomposition.tchebicheff import Tchebicheff


def weighted_sum(weights: np.ndarray, reward: np.ndarray) -> float:
    return np.dot(weights, reward)


def tchebicheff(tau: float, reward_dim: int):
    """
    Tchebicheff with automatically adapting utopian point.
    The ref point is always the best value seen so far (in each dimension) + the constant tau.
    :param: tau a constant to add to each best objective value
    """
    best_so_far = [float("-inf") for _ in range(reward_dim)]
    tch = Tchebicheff()

    def thunk(weights: np.ndarray, reward: np.ndarray):
        for i, r in enumerate(reward):
            if best_so_far[i] < r + tau:
                best_so_far[i] = r + tau
                print(f"Improved ref point {best_so_far}")
        # Losses have to be positive for gradient descent in EUPG, hence 1/distance instead of -distance
        return 1 / tch.do(F=reward, weights=weights, utopian_point=np.array(best_so_far))[0][0]

    return thunk

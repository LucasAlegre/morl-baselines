import numpy as np
from pymoo.decomposition.tchebicheff import Tchebicheff


def weighted_sum(reward: np.ndarray, weights: np.ndarray) -> float:
    return np.dot(reward, weights)


def tchebicheff(tau: float, reward_dim: int):
    best_so_far = [float("-inf") for _ in range(reward_dim)]

    def thunk(reward: np.ndarray, weights: np.ndarray):
        for i, r in enumerate(reward):
            if best_so_far[i] < r + tau:
                best_so_far[i] = r + tau
        tch = Tchebicheff()
        return -tch.do(F=reward, weights=weights, utopian_point=np.array(best_so_far))[0][0]

    return thunk

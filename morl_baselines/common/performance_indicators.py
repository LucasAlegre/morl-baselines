"""Performance indicators for multi-objective RL algorithms.

We mostly rely on pymoo for the computation of axiomatic indicators (HV and IGD), but some are customly made.
"""
from copy import deepcopy
from typing import Callable, List

import numpy as np
import numpy.typing as npt
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD


def hypervolume(ref_point: np.ndarray, points: List[npt.ArrayLike]) -> float:
    """Computes the hypervolume metric for a set of points (value vectors) and a reference point (from Pymoo).

    Args:
        ref_point (np.ndarray): Reference point
        points (List[np.ndarray]): List of value vectors

    Returns:
        float: Hypervolume metric
    """
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)


def igd(known_front: List[np.ndarray], current_estimate: List[np.ndarray]) -> float:
    """Inverted generational distance metric. Requires to know the optimal front.

    Args:
        known_front: known pareto front for the problem
        current_estimate: current pareto front

    Return:
        a float stating the average distance between a point in current_estimate and its nearest point in known_front
    """
    ind = IGD(np.array(known_front))
    return ind(np.array(current_estimate))


def sparsity(front: List[np.ndarray]) -> float:
    """Sparsity metric from PGMORL.

    Basically, the sparsity is the average distance between each point in the front.

    Args:
        front: current pareto front to compute the sparsity on

    Returns:
        float: sparsity metric
    """
    if len(front) < 2:
        return 0.0

    sparsity_value = 0.0
    m = len(front[0])
    front = np.array(front)
    for dim in range(m):
        objs_i = np.sort(deepcopy(front.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity_value += np.square(objs_i[i] - objs_i[i - 1])
    sparsity_value /= len(front) - 1

    return sparsity_value


def expected_utility(front: List[np.ndarray], weights_set: List[np.ndarray], utility: Callable = np.dot) -> float:
    """Expected Utility Metric.

    Expected utility of the policies on the PF for various weights.
    Similar to R-Metrics in MOO. But only needs one PF approximation.
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Args:
        front: current pareto front to compute the eum on
        weights_set: weights to use for the utility computation
        utility: utility function to use (default: dot product)

    Returns:
        float: eum metric
    """
    maxs = []
    for weights in weights_set:
        scalarized_front = np.array([utility(weights, point) for point in front])
        maxs.append(np.max(scalarized_front))

    return np.mean(np.array(maxs), axis=0)


def cardinality(front: List[np.ndarray]) -> float:
    """Cardinality Metric.

    Cardinality of the Pareto front approximation.

    Args:
        front: current pareto front to compute the cardinality on

    Returns:
        float: cardinality metric
    """
    return len(front)


def maximum_utility_loss(
    front: List[np.ndarray], reference_set: List[np.ndarray], weights_set: np.ndarray, utility: Callable = np.dot
) -> float:
    """Maximum Utility Loss Metric.

    Maximum utility loss of the policies on the PF for various weights.
    Paper: L. M. Zintgraf, T. V. Kanters, D. M. Roijers, F. A. Oliehoek, and P. Beau, “Quality Assessment of MORL Algorithms: A Utility-Based Approach,” 2015.

    Args:
        front: current pareto front to compute the mul on
        reference_set: reference set (e.g. true Pareto front) to compute the mul on
        weights_set: weights to use for the utility computation
        utility: utility function to use (default: dot product)

    Returns:
        float: mul metric
    """
    max_scalarized_values_ref = [np.max([utility(weight, point) for point in reference_set]) for weight in weights_set]
    max_scalarized_values = [np.max([utility(weight, point) for point in front]) for weight in weights_set]
    utility_losses = [max_scalarized_values_ref[i] - max_scalarized_values[i] for i in range(len(max_scalarized_values))]
    return np.max(utility_losses)

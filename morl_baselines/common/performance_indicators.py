from copy import deepcopy
from typing import List

import numpy as np
from pymoo.factory import get_performance_indicator
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD


def hypervolume(ref_point: np.ndarray, points: List[np.ndarray]) -> float:
    """Computes the hypervolume metric for a set of points (value vectors) and a reference point.

    Args:
        ref_point (np.ndarray): Reference point
        points (List[np.ndarray]): List of value vectors

    Returns:
        float: Hypervolume metric
    """
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)


def sparsity(front: List[np.ndarray]) -> float:
    """
    Sparsity metric from PGMORL
    :param front: current pareto front to compute the sparsity on
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


def igd(known_front: List[np.ndarray], current_estimate: List[np.ndarray]) -> float:
    """
    Inverted generational distance metric. Requires to know the front.
    :param known_front: known pareto front for the problem
    :param current_estimate: current pareto front
    :return: a float stating the average distance between a point in current_estimate and its nearest point in known_front
    """
    ind = IGD(np.array(known_front))
    return ind(np.array(current_estimate))

from copy import deepcopy
from typing import List

import numpy as np
from pymoo.factory import get_performance_indicator


# TODO should this be here or in MO-gym?
def hypervolume(ref_point: np.ndarray, points: List[np.ndarray]) -> float:
    """Computes the hypervolume metric for a set of points (value vectors) and a reference point.

    Args:
        ref_point (np.ndarray): Reference point
        points (List[np.ndarray]): List of value vectors

    Returns:
        float: Hypervolume metric
    """
    hv = get_performance_indicator("hv", ref_point=ref_point * -1)
    return hv.do(np.array(points) * -1)


def sparsity(front: List[np.ndarray]) -> float:
    """
    Sparsity metric from PGMORL
    :param front: current pareto front to compute the sparsity on
    """
    if len(front) < 2:
        return 0.

    sparsity_value = 0.
    m = len(front[0])
    front = np.array(front)
    for dim in range(m):
        objs_i = np.sort(deepcopy(front.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity_value += np.square(objs_i[i] - objs_i[i - 1])
    sparsity_value /= (len(front) - 1)

    return sparsity_value

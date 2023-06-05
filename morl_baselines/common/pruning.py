import numpy as np
from scipy.spatial import ConvexHull


def fast_c_prune(candidates):
    """A fast version to prune a set of points to its convex hull. This leverages the QuickHull algorithm.

    This algorithm first computes the convex hull of the set of points and then prunes the Pareto dominated points.

    Args:
        candidates (ndarray): A numpy array of vectors.

    Returns:
        ndarray: A convex coverage set.
    """
    hull = ConvexHull(candidates)
    ccs = candidates[hull.vertices]
    return fast_p_prune(ccs)


def arg_p_prune(candidates, remove_duplicates=True):
    """A batched and fast version of the Pareto coverage set algorithm.

    Args:
        candidates (ndarray): A numpy array of vectors.
        remove_duplicates (bool, optional): Whether to remove duplicate vectors. Defaults to True.

    Returns:
        ndarray: The indices of the elements that should be kept to form the Pareto front or coverage set.
    """
    if len(candidates) == 1:
        return candidates

    uniques, indcs, invs, counts = np.unique(candidates, return_index=True, return_inverse=True, return_counts=True,
                                             axis=0)

    res_eq = np.all(candidates[:, None, None] <= candidates, axis=-1).squeeze()
    res_g = np.all(candidates[:, None, None] < candidates, axis=-1).squeeze()
    c1 = np.sum(res_eq, axis=-1) == counts[invs]
    c2 = np.any(~res_g, axis=-1)
    if remove_duplicates:
        to_keep = np.zeros(len(candidates), dtype=bool)
        to_keep[indcs] = 1
    else:
        to_keep = np.ones(len(candidates), dtype=bool)

    return np.logical_and(c1, c2) & to_keep


def fast_p_prune(candidates, remove_duplicates=True):
    """A batched and fast version of the Pareto coverage set algorithm.

    Args:
        candidates (ndarray): A numpy array of vectors.
        remove_duplicates (bool, optional): Whether to remove duplicate vectors. Defaults to True.

    Returns:
        ndarray: A Pareto coverage set.
    """
    return candidates[arg_p_prune(candidates, remove_duplicates=remove_duplicates)]

"""Pareto utilities."""
from copy import deepcopy
from typing import List, Union

import numpy as np
from scipy.spatial import ConvexHull


def get_non_pareto_dominated_inds(candidates: Union[np.ndarray, List], remove_duplicates: bool = True) -> np.ndarray:
    """A batched and fast version of the Pareto coverage set algorithm.

    Args:
        candidates (ndarray): A numpy array of vectors.
        remove_duplicates (bool, optional): Whether to remove duplicate vectors. Defaults to True.

    Returns:
        ndarray: The indices of the elements that should be kept to form the Pareto front or coverage set.
    """
    candidates = np.array(candidates)
    uniques, indcs, invs, counts = np.unique(candidates, return_index=True, return_inverse=True, return_counts=True, axis=0)

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


def filter_pareto_dominated(candidates: Union[np.ndarray, List], remove_duplicates: bool = True) -> np.ndarray:
    """A batched and fast version of the Pareto coverage set algorithm.

    Args:
        candidates (ndarray): A numpy array of vectors.
        remove_duplicates (bool, optional): Whether to remove duplicate vectors. Defaults to True.

    Returns:
        ndarray: A Pareto coverage set.
    """
    candidates = np.array(candidates)
    if len(candidates) < 2:
        return candidates
    return candidates[get_non_pareto_dominated_inds(candidates, remove_duplicates=remove_duplicates)]


def filter_convex_dominated(candidates: Union[np.ndarray, List]) -> np.ndarray:
    """A fast version to prune a set of points to its convex hull. This leverages the QuickHull algorithm.

    This algorithm first computes the convex hull of the set of points and then prunes the Pareto dominated points.

    Args:
        candidates (ndarray): A numpy array of vectors.

    Returns:
        ndarray: A convex coverage set.
    """
    candidates = np.array(candidates)
    if len(candidates) > 2:
        hull = ConvexHull(candidates)
        ccs = candidates[hull.vertices]
    else:
        ccs = candidates
    return filter_pareto_dominated(ccs)


def get_non_dominated(candidates: set) -> set:
    """This function returns the non-dominated subset of elements.

    Source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    The code provided in all the stackoverflow answers is wrong. Important changes have been made in this function.

    Args:
        candidates: The input set of candidate vectors.

    Returns:
        The non-dominated subset of this input set.
    """
    candidates = np.array(list(candidates))  # Turn the input set into a numpy array.
    candidates = candidates[candidates.sum(1).argsort()[::-1]]  # Sort candidates by decreasing sum of coordinates.
    for i in range(candidates.shape[0]):  # Process each point in turn.
        n = candidates.shape[0]  # Check current size of the candidates.
        if i >= n:  # If we've eliminated everything up until this size we stop.
            break
        non_dominated = np.ones(candidates.shape[0], dtype=bool)  # Initialize a boolean mask for undominated points.
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        non_dominated[i + 1 :] = np.any(candidates[i + 1 :] > candidates[i], axis=1)
        candidates = candidates[non_dominated]  # Grab only the non-dominated vectors using the generated bitmask.

    non_dominated = set()
    for candidate in candidates:
        non_dominated.add(tuple(candidate))  # Add the non dominated vectors to a set again.

    return non_dominated


def get_non_dominated_inds(solutions: np.ndarray) -> np.ndarray:
    """Returns a boolean array indicating which points are non-dominated."""
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1
    return is_efficient


class ParetoArchive:
    """Pareto archive."""

    def __init__(self, convex_hull: bool = False):
        """Initializes the Pareto archive."""
        self.convex_hull = convex_hull
        self.individuals: list = []
        self.evaluations: List[np.ndarray] = []

    def add(self, candidate, evaluation: np.ndarray):
        """Adds the candidate to the memory and removes Pareto inefficient points.

        Args:
            candidate: The candidate to add.
            evaluation: The evaluation of the candidate.
        """
        self.evaluations.append(evaluation)
        self.individuals.append(deepcopy(candidate))

        # Non-dominated sorting
        if self.convex_hull:
            nd_candidates = {tuple(x) for x in filter_convex_dominated(self.evaluations)}
        else:
            nd_candidates = {tuple(x) for x in filter_pareto_dominated(self.evaluations)}

        # Reconstruct the pareto archive (because Non-Dominated sorting might change the order of candidates)
        non_dominated_evals = []
        non_dominated_evals_tuples = []
        non_dominated_individuals = []
        for e, i in zip(self.evaluations, self.individuals):
            if tuple(e) in nd_candidates and tuple(e) not in non_dominated_evals_tuples:
                non_dominated_evals.append(e)
                non_dominated_evals_tuples.append(tuple(e))
                non_dominated_individuals.append(i)
        self.evaluations = non_dominated_evals
        self.individuals = non_dominated_individuals

"""Pareto utilities."""
from copy import deepcopy
from typing import List

import numpy as np


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

    def __init__(self):
        """Initializes the Pareto archive."""
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
        nd_candidates = get_non_dominated({tuple(e) for e in self.evaluations})

        # Reconstruct the pareto archive (because Non-Dominated sorting might change the order of candidates)
        non_dominated_evals = []
        non_dominated_individuals = []
        for e, i in zip(self.evaluations, self.individuals):
            if tuple(e) in nd_candidates:
                non_dominated_evals.append(e)
                non_dominated_individuals.append(i)
        self.evaluations = non_dominated_evals
        self.individuals = non_dominated_individuals

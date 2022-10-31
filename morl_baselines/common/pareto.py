from copy import deepcopy
from typing import List

import numpy as np


def get_non_dominated(candidates: set):
    """
    This function returns the non-dominated subset of elements.
    :param candidates: The input set of candidate vectors.
    :return: The non-dominated subset of this input set.
    Source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    The code provided in all the stackoverflow answers is wrong. Important changes have been made in this function.
    """
    candidates = np.array(list(candidates))  # Turn the input set into a numpy array.
    candidates = candidates[
        candidates.sum(1).argsort()[::-1]
    ]  # Sort candidates by decreasing sum of coordinates.
    for i in range(candidates.shape[0]):  # Process each point in turn.
        n = candidates.shape[0]  # Check current size of the candidates.
        if i >= n:  # If we've eliminated everything up until this size we stop.
            break
        nd = np.ones(
            candidates.shape[0], dtype=bool
        )  # Initialize a boolean mask for undominated points.
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        nd[i + 1 :] = np.any(candidates[i + 1 :] > candidates[i], axis=1)
        candidates = candidates[
            nd
        ]  # Grab only the non-dominated vectors using the generated bitmask.

    non_dominated = set()
    for candidate in candidates:
        non_dominated.add(
            tuple(candidate)
        )  # Add the non dominated vectors to a set again.

    return non_dominated


class ParetoArchive:
    def __init__(self):
        self.individuals: list = []
        self.evaluations: List[np.ndarray] = []

    def add(self, candidate, evaluation: np.ndarray):
        """
        Adds the candidate to the memory and removes Pareto inefficient points
        """
        self.evaluations.append(evaluation)
        self.individuals.append(deepcopy(candidate))
        # ND sorting
        nd_candidates = get_non_dominated(set([tuple(e) for e in self.evaluations]))

        # Reconstruct the pareto archive (because ND sorting might change the order of candidates)
        nd_evals = []
        nd_individuals = []
        for e, i in zip(self.evaluations, self.individuals):
            if tuple(e) in nd_candidates:
                nd_evals.append(e)
                nd_individuals.append(i)
        self.evaluations = nd_evals
        self.individuals = nd_individuals

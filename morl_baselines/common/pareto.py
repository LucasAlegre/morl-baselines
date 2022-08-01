from copy import deepcopy

import numpy as np


# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
import torch


def is_pareto_efficient(evaluations: np.ndarray, return_mask: bool = True):
    """
    Find the pareto-efficient points (maximization is supposed)
    :param evaluations: An (n_points, n_objectives) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(evaluations.shape[0])
    n_points = evaluations.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(evaluations):
        # Check for points which are better or equal to the current one
        nondominated_point_mask = np.any(evaluations > evaluations[next_point_index], axis=1)
        efficient_points = np.where(np.all(evaluations == evaluations[next_point_index], axis=1))
        nondominated_point_mask[efficient_points] = True

        # Apply mask to filter out the dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        evaluations = evaluations[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


class ParetoArchive:
    def __init__(self):
        self.individuals = []
        self.evaluations = []

    def add(self, candidate, evaluation: np.ndarray):
        """
        Adds the candidate to the memory if it is pareto efficient
        """
        self.evaluations.append(evaluation)
        self.individuals.append(deepcopy(candidate))
        mask = is_pareto_efficient(np.array([np.array(e) for e in self.evaluations]))
        self.evaluations = np.array(self.evaluations)[mask].tolist()
        self.individuals = np.array(self.individuals)[mask].tolist()

"""Linear Support implementation."""

import random
from copy import deepcopy
from typing import List, Optional

import cdd
import cvxpy as cp
import numpy as np
from cvxpy import SolverError
from gymnasium.core import Env

from morl_baselines.common.evaluation import policy_evaluation_mo
from morl_baselines.common.morl_algorithm import MOPolicy
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.common.weights import extrema_weights


np.set_printoptions(precision=4)


class LinearSupport:
    """Linear Support for computing corner weights when using linear utility functions.

    Implements both

    Optimistic Linear Support (OLS) algorithm:
    Paper: (Section 3.3 of http://roijers.info/pub/thesis.pdf).

    Generalized Policy Improvement Linear Support (GPI-LS) algorithm:
    Paper: https://arxiv.org/abs/2301.07784
    """

    def __init__(
        self,
        num_objectives: int,
        epsilon: float = 0.0,
        verbose: bool = True,
    ):
        """Initialize Linear Support.

        Args:
            num_objectives (int): Number of objectives
            epsilon (float, optional): Minimum improvement per iteration. Defaults to 0.0.
            verbose (bool): Defaults to False.
        """
        self.num_objectives = num_objectives
        self.epsilon = epsilon
        self.visited_weights = []  # List of already tested weight vectors
        self.ccs = []
        self.weight_support = []  # List of weight vectors for each value vector in the CCS
        self.queue = []
        self.iteration = 0
        self.ols_ended = False
        self.verbose = verbose
        for w in extrema_weights(self.num_objectives):
            self.queue.append((float("inf"), w))

    def next_weight(
        self, algo: str = "ols", gpi_agent: Optional[MOPolicy] = None, env: Optional[Env] = None, rep_eval: int = 1
    ) -> np.ndarray:
        """Returns the next weight vector with highest priority.

        Args:
            algo (str): Algorithm to use. Either 'ols' or 'gpi-ls'.
            gpi_agent (Optional[MOPolicy]): Agent to use for GPI-LS.
            env (Optional[Env]): Environment to use for GPI-LS.
            rep_eval (int): Number of times to evaluate the agent in GPI-LS.

        Returns:
            np.ndarray: Next weight vector
        """
        if len(self.ccs) > 0:
            W_corner = self.compute_corner_weights()
            if self.verbose:
                print("W_corner:", W_corner, "W_corner size:", len(W_corner))

            self.queue = []
            for wc in W_corner:
                if algo == "ols":
                    priority = self.ols_priority(wc)

                elif algo == "gpi-ls":
                    if gpi_agent is None:
                        raise ValueError("GPI-LS requires passing a GPI agent.")
                    gpi_expanded_set = [policy_evaluation_mo(gpi_agent, env, wc, rep=rep_eval)[3] for wc in W_corner]
                    priority = self.gpi_ls_priority(wc, gpi_expanded_set)

                if self.epsilon is None or priority >= self.epsilon:
                    # OLS does not try the same weight vector twice
                    if not (algo == "ols" and any([np.allclose(wc, wv) for wv in self.visited_weights])):
                        self.queue.append((priority, wc))

            if len(self.queue) > 0:
                # Sort in descending order of priority
                self.queue.sort(key=lambda t: t[0], reverse=True)
                # If all priorities are 0, shuffle the queue to avoid repearting weights every iteration
                if self.queue[0][0] == 0.0:
                    random.shuffle(self.queue)

        if self.verbose:
            print("CCS:", self.ccs, "CCS size:", len(self.ccs))

        if len(self.queue) == 0:
            if self.verbose:
                print("There are no corner weights in the queue. Returning None.")
            self.ols_ended = True
            return None
        else:
            next_w = self.queue.pop(0)[1]
            if self.verbose:
                print("Next weight:", next_w)
            return next_w

    def get_weight_support(self) -> List[np.ndarray]:
        """Returns the weight support of the CCS.

        Returns:
            List[np.ndarray]: List of weight vectors of the CCS

        """
        return deepcopy(self.weight_support)

    def get_corner_weights(self, top_k: Optional[int] = None) -> List[np.ndarray]:
        """Returns the corner weights of the current CCS.

        Args:
            top_k: If not None, returns the top_k corner weights.

        Returns:
            List[np.ndarray]: List of corner weights.
        """
        weights = [w.copy() for (p, w) in self.queue]
        if top_k is not None:
            return weights[:top_k]
        else:
            return weights

    def ended(self) -> bool:
        """Returns True if there are no more corner weights to test.

        Warning: This method must be called AFTER calling next_weight().
        Ex: w = ols.next_weight()
            if ols.ended():
                print("OLS ended.")
        """
        return self.ols_ended

    def add_solution(self, value: np.ndarray, w: np.ndarray) -> List[int]:
        """Add new value vector optimal to weight w.

        Args:
            value (np.ndarray): New value vector
            w (np.ndarray): Weight vector

        Returns:
            List of indices of value vectors removed from the CCS for being dominated.
        """
        if self.verbose:
            print(f"Adding value: {value} to CCS.")

        self.iteration += 1
        self.visited_weights.append(w)

        if self.is_dominated(value):
            if self.verbose:
                print(f"Value {value} is dominated. Discarding.")
            return [len(self.ccs)]

        removed_indx = self.remove_obsolete_values(value)

        self.ccs.append(value)
        self.weight_support.append(w)

        return removed_indx

    def ols_priority(self, w: np.ndarray) -> float:
        """Get the priority of a weight vector for OLS.

        Args:
            w: Weight vector

        Returns:
            Priority of the weight vector.
        """
        max_value_ccs = self.max_scalarized_value(w)
        max_optimistic_value = self.max_value_lp(w)
        priority = max_optimistic_value - max_value_ccs
        return priority

    def gpi_ls_priority(self, w: np.ndarray, gpi_expanded_set: List[np.ndarray]) -> float:
        """Get the priority of a weight vector for GPI-LS.

        Args:
            w: Weight vector

        Returns:
            Priority of the weight vector.
        """

        def best_vector(values, w):
            max_v = values[0]
            for i in range(1, len(values)):
                if values[i] @ w > max_v @ w:
                    max_v = values[i]
            return max_v

        max_value_ccs = self.max_scalarized_value(w)
        max_value_gpi = best_vector(gpi_expanded_set, w)
        max_value_gpi = np.dot(max_value_gpi, w)
        priority = max_value_gpi - max_value_ccs

        return priority

    def max_scalarized_value(self, w: np.ndarray) -> Optional[float]:
        """Returns the maximum scalarized value for weight vector w.

        Args:
            w: Weight vector

        Returns:
            Maximum scalarized value for weight vector w.
        """
        if len(self.ccs) == 0:
            return None
        return np.max([np.dot(v, w) for v in self.ccs])

    def remove_obsolete_weights(self, new_value: np.ndarray) -> List[np.ndarray]:
        """Remove from the queue the weight vectors for which the new value vector is better than previous values.

        Args:
            new_value: New value vector

        Returns:
            List of weight vectors removed from the queue.
        """
        if len(self.ccs) == 0:
            return []
        W_del = []
        inds_remove = []
        for i, (priority, cw) in enumerate(self.queue):
            if np.dot(cw, new_value) > self.max_scalarized_value(cw):
                W_del.append(cw)
                inds_remove.append(i)
        for i in reversed(inds_remove):
            self.queue.pop(i)
        return W_del

    def remove_obsolete_values(self, value: np.ndarray) -> List[int]:
        """Removes the values vectors which are no longer optimal for any weight vector after adding the new value vector.

        Args:
            value (np.ndarray): New value vector

        Returns:
            The indices of the removed values.
        """
        removed_indx = []
        for i in reversed(range(len(self.ccs))):
            weights_optimal = [
                w
                for w in self.visited_weights
                if np.dot(self.ccs[i], w) == self.max_scalarized_value(w) and np.dot(value, w) < np.dot(self.ccs[i], w)
            ]
            if len(weights_optimal) == 0:
                print("removed value", self.ccs[i])
                removed_indx.append(i)
                self.ccs.pop(i)
                self.weight_support.pop(i)
        return removed_indx

    def max_value_lp(self, w_new: np.ndarray) -> float:
        """Returns an upper-bound for the maximum value of the scalarized objective.

        Args:
            w_new: New weight vector

        Returns:
            Upper-bound for the maximum value of the scalarized objective.
        """
        # No upper bound if no values in CCS
        if len(self.ccs) == 0:
            return float("inf")

        w = cp.Parameter(self.num_objectives)
        w.value = w_new
        v = cp.Variable(self.num_objectives)

        W_ = np.vstack(self.visited_weights)
        W = cp.Parameter(W_.shape)
        W.value = W_

        V_ = np.array([self.max_scalarized_value(weight) for weight in self.visited_weights])
        V = cp.Parameter(V_.shape)
        V.value = V_

        # Maximum value for weight vector w
        objective = cp.Maximize(w @ v)
        # such that it is consistent with other optimal values for other visited weights
        constraints = [W @ v <= V]
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(verbose=False)
        except SolverError:
            print("ECOS solver error, trying another one.")
            result = prob.solve(solver=cp.SCS, verbose=False)
        return result

    def compute_corner_weights(self) -> List[np.ndarray]:
        """Returns the corner weights for the current set of values.

        See http://roijers.info/pub/thesis.pdf Definition 19.
        Obs: there is a typo in the definition of the corner weights in the thesis, the >= sign should be <=.

        Returns:
            List of corner weights.
        """
        A = np.vstack(self.ccs)
        A = np.round_(A, decimals=4)  # Round to avoid numerical issues
        A = np.concatenate((A, -np.ones(A.shape[0]).reshape(-1, 1)), axis=1)

        A_plus = np.ones(A.shape[1]).reshape(1, -1)
        A_plus[0, -1] = 0
        A = np.concatenate((A, A_plus), axis=0)
        A_plus = -np.ones(A.shape[1]).reshape(1, -1)
        A_plus[0, -1] = 0
        A = np.concatenate((A, A_plus), axis=0)

        for i in range(self.num_objectives):
            A_plus = np.zeros(A.shape[1]).reshape(1, -1)
            A_plus[0, i] = -1
            A = np.concatenate((A, A_plus), axis=0)

        b = np.zeros(len(self.ccs) + 2 + self.num_objectives)
        b[len(self.ccs)] = 1
        b[len(self.ccs) + 1] = -1

        def compute_poly_vertices(A, b):
            # Based on https://stackoverflow.com/questions/65343771/solve-linear-inequalities
            b = b.reshape((b.shape[0], 1))
            mat = cdd.Matrix(np.hstack([b, -A]), number_type="float")
            mat.rep_type = cdd.RepType.INEQUALITY
            P = cdd.Polyhedron(mat)
            g = P.get_generators()
            V = np.array(g)
            vertices = []
            for i in range(V.shape[0]):
                if V[i, 0] != 1:
                    continue
                if i not in g.lin_set:
                    vertices.append(V[i, 1:])
            return vertices

        vertices = compute_poly_vertices(A, b)
        corners = []
        for v in vertices:
            corner_weight = v[:-1]
            # Make sure the corner weight is positive and sum to 1
            corner_weight = np.abs(corner_weight)
            corner_weight /= corner_weight.sum()
            corners.append(corner_weight)

        return corners

    def is_dominated(self, value: np.ndarray) -> bool:
        """Checks if the value is dominated by any of the values in the CCS.

        Args:
            value: Value vector

        Returns:
            True if the value is dominated by any of the values in the CCS, False otherwise.
        """
        if len(self.ccs) == 0:
            return False
        for w in self.visited_weights:
            if np.dot(value, w) >= self.max_scalarized_value(w):
                return False
        return True


if __name__ == "__main__":

    def _solve(w):
        return np.array(list(map(float, input().split())), dtype=np.float32)

    num_objectives = 2
    ols = LinearSupport(num_objectives=num_objectives, epsilon=0.0001, verbose=True)
    w = ols.next_weight()
    while not ols.ended():
        print("w:", w)
        value = _solve(w)
        ols.add_solution(value, w)

        print("hv:", hypervolume(np.zeros(num_objectives), ols.ccs))
        w = ols.next_weight()

"""OLS implementation."""
from copy import deepcopy
from typing import List, Optional

import cdd
import cvxpy as cp
import numpy as np

from morl_baselines.common.performance_indicators import hypervolume


np.set_printoptions(precision=4)


class OLS:
    """Optimistic Linear Support.

    Outer loop method to select next weight vector.
    Paper: (Section 3.3 of http://roijers.info/pub/thesis.pdf).
    """

    def __init__(
        self,
        num_objectives: int,
        epsilon: float = 0.0,
        verbose: bool = False,
    ):
        """Initialize OLS.

        Args:
            num_objectives (int): Number of objectives
            epsilon (float, optional): Minimum improvement per iteration. Defaults to 0.0.
            verbose (bool): Defaults to False.
        """
        self.num_objectives = num_objectives
        self.epsilon = epsilon
        self.visited_weights = []  # List of already tested weight vectors
        self.ccs = []
        self.ccs_weights = []  # List of weight vectors for each value vector in the CCS
        self.queue = []
        self.iteration = 0
        self.verbose = verbose
        for w in self.extrema_weights():
            self.queue.append((float("inf"), w))

    def next_weight(self) -> np.ndarray:
        """Returns the next weight vector with highest priority.

        Returns:
            np.ndarray: Next weight vector
        """
        return self.queue.pop(0)[1]

    def get_ccs_weights(self) -> List[np.ndarray]:
        """Returns the weights in the CCS.

        Returns:
            List[np.ndarray]: List of weight vectors in the CCS

        """
        return deepcopy(self.ccs_weights)

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
        """Returns True if the queue is empty."""
        return len(self.queue) == 0

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
                print("Value is dominated. Discarding.")
            return [len(self.ccs)]

        W_del = self.remove_obsolete_weights(new_value=value)
        W_del.append(w)

        removed_indx = self.remove_obsolete_values(value)

        self.ccs.append(value)
        self.ccs_weights.append(w)

        W_corner = self.compute_corner_weights()

        self.queue.clear()
        for wc in W_corner:
            priority = self.get_priority(wc)
            if priority > self.epsilon:
                if self.verbose:
                    print(f"Adding weight: {wc} to queue with priority {priority}.")
                self.queue.append((priority, wc))
        self.queue.sort(key=lambda t: t[0], reverse=True)  # Sort in descending order of priority

        if self.verbose:
            print(f"CCS: {self.ccs}")
            print(f"CCS size: {len(self.ccs)}")

        return removed_indx

    def get_priority(self, w: np.ndarray) -> float:
        """Get the priority of a weight vector.

        Args:
            w: Weight vector

        Returns:
            Priority of the weight vector.
        """
        max_optimistic_value = self.max_value_lp(w)
        max_value_ccs = self.max_scalarized_value(w)
        priority = max_optimistic_value - max_value_ccs  # / abs(max_optimistic_value)
        return priority

    def max_scalarized_value(self, w: np.ndarray) -> Optional[float]:
        """Returns the maximum scalarized value for weight vector w.

        Args:
            w: Weight vector

        Returns:
            Maximum scalarized value for weight vector w.
        """
        if not self.ccs:
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
        """Removes the values vectors which are dominated by the new value for all visited weight vectors.

        Args:
            value: New value vector

        Returns:
             the indices of the removed values.
        """
        removed_indx = []
        for i in reversed(range(len(self.ccs))):
            best_in_all = True
            for j in range(len(self.visited_weights)):
                w = self.visited_weights[j]
                if np.dot(value, w) < np.dot(self.ccs[i], w):
                    best_in_all = False
                    break

            if best_in_all:
                removed_indx.append(i)
                self.ccs.pop(i)
                self.ccs_weights.pop(i)

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
        return prob.solve(verbose=False)

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
            corners.append(v[:-1])

        # Do not include corner weights already in visited weights
        def predicate(wc):
            return (wc is not None) and (not any([np.allclose(wc, w_old) for w_old in self.visited_weights]))

        corners = list(filter(predicate, corners))
        return corners

    def extrema_weights(self) -> List[np.ndarray]:
        """Returns the weight vectors which have one component equal to 1 and the rest equal to 0.

        Returns:
            List of weight vectors.
        """
        extrema_weights = []
        for i in range(self.num_objectives):
            w = np.zeros(self.num_objectives)
            w[i] = 1.0
            extrema_weights.append(w)
        return extrema_weights

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

    num_objectives = 3
    ols = OLS(num_objectives=num_objectives, epsilon=0.0001, verbose=True)  # , min_value=0.0, max_value=1 / (1 - 0.95) * 1)
    while not ols.ended():
        w = ols.next_weight()
        print("w:", w)
        value = _solve(w)
        ols.add_solution(value, w)

        print("hv:", hypervolume(np.zeros(num_objectives), ols.ccs))

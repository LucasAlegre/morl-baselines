from itertools import combinations
from typing import List, Optional

import cvxpy as cp
import numpy as np

from morl_baselines.common.performance_indicators import hypervolume

np.set_printoptions(precision=4)


class OLS:
    """ Optimistic Linear Support (Section 3.3 of http://roijers.info/pub/thesis.pdf)
        Outer loop method to select next weight vector

        Args:
            num_objectives (int): Number of objectives
            epsilon (float, optional): Minimum improvement per iteration. Defaults to 0.0.
            verbose (bool): Defaults to False.
    """
    def __init__(
        self,
        num_objectives: int,
        epsilon: float = 0.0,
        verbose: bool = False,
    ):
        self.num_objectives = num_objectives
        self.epsilon = epsilon
        self.W = []
        self.ccs = []
        self.ccs_weights = []
        self.queue = []
        self.iteration = 0
        self.verbose = verbose
        for w in self.extrema_weights():
            self.queue.append((float("inf"), w))

    def next_weight(self) -> np.ndarray:
        return self.queue.pop(0)[1]

    def get_ccs_weights(self) -> List[np.ndarray]:
        return self.ccs_weights.copy()

    def get_corner_weights(self, top_k: Optional[int] = None) -> List[np.ndarray]:
        weights = [w for (p, w) in self.queue]
        if top_k is not None:
            return weights[:top_k]
        else:
            return weights

    def ended(self) -> bool:
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
        self.W.append(w)
        if self.is_dominated_or_equal(value):
            if self.verbose:
                print("Value is dominated. Discarding.")
            return [len(self.ccs)]

        W_del = self.remove_obsolete_weights(new_value=value)
        W_del.append(w)

        removed_indx = self.remove_obsolete_values(value)

        W_corner = self.new_corner_weights(value, W_del)

        self.ccs.append(value)
        self.ccs_weights.append(w)

        print("W_corner", W_corner)
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

    def get_priority(self, w) -> float:
        max_optimistic_value = self.max_value_lp(w)
        max_value_ccs = self.max_scalarized_value(w)
        priority = max_optimistic_value - max_value_ccs  # / abs(max_optimistic_value)
        return priority

    def max_scalarized_value(self, w: np.ndarray) -> float:
        if not self.ccs:
            return None
        return np.max([np.dot(v, w) for v in self.ccs])

    def get_set_max_policy_index(self, w: np.ndarray) -> int:
        if not self.ccs:
            return None
        return np.argmax([np.dot(v, w) for v in self.ccs])

    def remove_obsolete_weights(self, new_value: np.ndarray) -> List[np.ndarray]:
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
        removed_indx = []
        for i in reversed(range(len(self.ccs))):
            best_in_all = True
            for j in range(len(self.W)):
                w = self.W[j]
                if np.dot(value, w) < np.dot(self.ccs[i], w):
                    best_in_all = False
                    break
            if best_in_all:
                removed_indx.append(i)
                self.ccs.pop(i)
                self.ccs_weights.pop(i)
        return removed_indx

    def max_value_lp(self, w_new: np.ndarray) -> float:
        if len(self.ccs) == 0:
            return float("inf")
        w = cp.Parameter(self.num_objectives)
        w.value = w_new
        v = cp.Variable(self.num_objectives)
        W_ = np.vstack(self.W)
        V_ = np.array([self.max_scalarized_value(weight) for weight in self.W])
        W = cp.Parameter(W_.shape)
        W.value = W_
        V = cp.Parameter(V_.shape)
        V.value = V_
        objective = cp.Maximize(w @ v)
        constraints = [W @ v <= V]
        prob = cp.Problem(objective, constraints)
        return prob.solve(verbose=False)

    def new_corner_weights(self, v_new: np.ndarray, W_del: List[np.ndarray]) -> List[np.ndarray]:
        if len(self.ccs) == 0:
            return []
        V_rel = []
        W_new = []
        for w in W_del:
            best = [self.ccs[0]]
            for v in self.ccs[1:]:
                if np.allclose(np.dot(w, v), np.dot(w, best[0])):
                    best.append(v)
                elif np.dot(w, v) > np.dot(w, best[0]):
                    best = [v]
            V_rel += best
            if len(best) < self.num_objectives:
                wc = self.corner_weight(v_new, best)
                W_new.append(wc)
                W_new.extend(self.extrema_weights())

        V_rel = np.unique(V_rel, axis=0)
        for comb in range(1, self.num_objectives):
            for x in combinations(V_rel, comb):
                if not x:
                    continue
                wc = self.corner_weight(v_new, x)
                W_new.append(wc)

        filter_fn = lambda wc: (wc is not None) and (not any([np.allclose(wc, w_old) for w_old in self.W] + [np.allclose(wc, w_old) for p, w_old in self.queue]))
        W_new = list(filter(filter_fn, W_new))
        W_new = np.unique(W_new, axis=0)
        return W_new

    def corner_weight(self, v_new: np.ndarray, v_set: List[np.ndarray]) -> np.ndarray:
        wc = cp.Variable(self.num_objectives)
        v_n = cp.Parameter(self.num_objectives)
        v_n.value = v_new
        objective = cp.Minimize(v_n @ wc)  # cp.Minimize(0)
        constraints = [0 <= wc, cp.sum(wc) == 1]
        for v in v_set:
            v_par = cp.Parameter(self.num_objectives)
            v_par.value = v
            constraints.append(v_par @ wc == v_n @ wc)
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)  # (solver='SCS', verbose=False, eps=1e-5)
        if prob.status == cp.OPTIMAL:
            weight = np.clip(wc.value, 0, 1)  # ensure range [0,1]
            weight /= weight.sum()  # ensure sum to one
            return weight
        else:
            return None

    def extrema_weights(self) -> List[np.ndarray]:
        extrema_weights = []
        for i in range(self.num_objectives):
            w = np.zeros(self.num_objectives)
            w[i] = 1.0
            extrema_weights.append(w)
        return extrema_weights

    def is_dominated_or_equal(self, value: np.ndarray) -> bool:
        for v in self.ccs:
            if ((v >= value).all() and (v > value).any()) or np.allclose(v, value):
                return True
        return False


if __name__ == "__main__":

    def solve(w):
        return np.array(list(map(float, input().split())), dtype=np.float32)

    num_objectives = 4
    ols = OLS(num_objectives=num_objectives, epsilon=0.0001) #, min_value=0.0, max_value=1 / (1 - 0.95) * 1)
    while not ols.ended():
        w = ols.next_w()
        print("w:", w)
        value = solve(w)
        ols.add_solution(value, w)

        print("hv:", hypervolume(np.zeros(num_objectives), ols.ccs))

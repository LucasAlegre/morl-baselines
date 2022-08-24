from itertools import combinations
from typing import List, Optional

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

import wandb as wb

from morl_baselines.common.performance_indicators import hypervolume

np.set_printoptions(precision=4)


class OLS:
    # Section 3.3 of http://roijers.info/pub/thesis.pdf
    def __init__(
        self,
        m: int,
        epsilon: float = 0.0,
        negative_weights: bool = False,
        max_value: Optional[float] = None,
        min_value: Optional[float] = None,
        reverse_extremum: bool = False,
    ):
        self.m = m
        self.epsilon = epsilon
        self.W = []
        self.ccs = []
        self.ccs_weights = []
        self.queue = []
        self.iteration = 0
        self.max_value = max_value
        self.min_value = min_value
        self.negative_weights = negative_weights
        self.worst_case_weight_repeated = False
        extremum_weights = reversed(self.extrema_weights()) if reverse_extremum else self.extrema_weights()
        for w in extremum_weights:
            self.queue.append((float("inf"), w))

    def next_w(self) -> np.ndarray:
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
        return len(self.queue) == 0 or self.worst_case_weight_repeated

    def add_solution(self, value, w, gpi_agent=None, env=None) -> int:
        print("value:", value)
        self.iteration += 1
        self.W.append(w)
        if self.is_dominated(value):
            return [len(self.ccs)]
        for i, v in enumerate(self.ccs):
            if np.allclose(v, value):
                return [len(self.ccs)]  # delete new policy as it has same value as an old one

        W_del = self.remove_obsolete_weights(new_value=value)
        W_del.append(w)
        print("W_del", W_del)

        removed_indx = self.remove_obsolete_values(value)

        W_corner = self.new_corner_weights(value, W_del)

        self.ccs.append(value)
        self.ccs_weights.append(w)

        print("W_corner", W_corner)
        for wc in W_corner:
            priority = self.get_priority(wc, gpi_agent, env)
            print("improv.", priority)
            if priority > self.epsilon:
                self.queue.append((priority, wc))
        self.queue.sort(key=lambda t: t[0], reverse=True)  # Sort in descending order of priority

        print("ccs:", self.ccs)
        print("ccs size:", len(self.ccs))

        return removed_indx

    def get_priority(self, w) -> float:
        max_optimistic_value = self.max_value_lp(w)
        max_value_ccs = self.max_scalarized_value(w)
        # upper_bound_nemecek = self.upper_bound_policy_caches(w)
        # print(f'optimistic: {max_optimistic_value} policy_cache_up: {upper_bound_nemecek}')

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
            if np.dot(cw, new_value) > self.max_scalarized_value(cw):  # and priority != float('inf'):
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
                print("removed value", self.ccs[i])
                removed_indx.append(i)
                self.ccs.pop(i)
                self.ccs_weights.pop(i)
        return removed_indx

    def max_value_lp(self, w_new: np.ndarray) -> float:
        if len(self.ccs) == 0:
            return float("inf")
        w = cp.Parameter(self.m)
        w.value = w_new
        v = cp.Variable(self.m)
        W_ = np.vstack(self.W)
        V_ = np.array([self.max_scalarized_value(weight) for weight in self.W])
        W = cp.Parameter(W_.shape)
        W.value = W_
        V = cp.Parameter(V_.shape)
        V.value = V_
        objective = cp.Maximize(w @ v)
        constraints = [W @ v <= V]
        if self.max_value is not None:
            constraints.append(v <= self.max_value)
        if self.min_value is not None:
            constraints.append(v >= self.min_value)
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
            if len(best) < self.m:
                wc = self.corner_weight(v_new, best)
                W_new.append(wc)
                W_new.extend(self.extrema_weights())

        V_rel = np.unique(V_rel, axis=0)
        # V_rel = self.ccs.copy()
        for comb in range(1, self.m):
            for x in combinations(V_rel, comb):
                if not x:
                    continue
                wc = self.corner_weight(v_new, x)
                W_new.append(wc)

        filter_fn = lambda wc: (wc is not None) and (not any([np.allclose(wc, w_old) for w_old in self.W] + [np.allclose(wc, w_old) for p, w_old in self.queue]))
        # (np.isclose(np.dot(wc, v_new), self.max_scalarized_value(wc))) and \
        W_new = list(filter(filter_fn, W_new))
        W_new = np.unique(W_new, axis=0)
        return W_new

    def corner_weight(self, v_new: np.ndarray, v_set: List[np.ndarray]) -> np.ndarray:
        wc = cp.Variable(self.m)
        v_n = cp.Parameter(self.m)
        v_n.value = v_new
        objective = cp.Minimize(v_n @ wc)  # cp.Minimize(0)
        constraints = [0 <= wc, cp.sum(wc) == 1]
        for v in v_set:
            v_par = cp.Parameter(self.m)
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
        for i in range(self.m):
            w = np.zeros(self.m)
            w[i] = 1.0
            extrema_weights.append(w)
        return extrema_weights

    def is_dominated(self, value):
        for v in self.ccs:
            if (v > value).all():
                return True
        return False


if __name__ == "__main__":

    def solve(w):
        return np.array(list(map(float, input().split())), dtype=np.float32)

    m = 4
    ols = OLS(m=m, epsilon=0.0001) #, min_value=0.0, max_value=1 / (1 - 0.95) * 1)
    while not ols.ended():
        w = ols.next_w()
        print("w:", w)
        value = solve(w)
        ols.add_solution(value, w)

        print("hv:", hypervolume(np.zeros(m), ols.ccs))

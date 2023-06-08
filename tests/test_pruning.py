import unittest

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import halfnorm

from morl_baselines.common.pareto import (
    filter_convex_dominated,
    filter_pareto_dominated,
)


def generate_known_front(num_nd_points, num_d_points, convex=False, dims=2, decimals=None, min_val=0, max_val=10, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    nd_points = sample_from_unit_ball_positive(
        num_nd_points, dims=dims, decimals=decimals, min_val=min_val, max_val=max_val, rng=rng
    )
    if convex:
        d_points = sample_convex_combinations(nd_points, num_nd_points, dims=dims, decimals=decimals, rng=rng)
    else:
        d_points = sample_dominated_points(nd_points, num_d_points, decimals=decimals, rng=rng)
    return nd_points, d_points


def sample_from_unit_ball_positive(num_points, dims=2, decimals=None, min_val=0, max_val=10, rng=None):
    points = halfnorm.rvs(size=(num_points, dims), random_state=rng)
    lambdas = np.sqrt(np.sum(points * points, axis=1))
    points = points / lambdas[:, None]
    points = points * (max_val - min_val) + min_val
    if decimals is not None:
        points = np.around(points, decimals)
    return points


def sample_convex_combinations(nd_points, num_points, dims=2, decimals=None, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    ch = ConvexHull(nd_points)
    simplices = ch.simplices
    parents = rng.choice(simplices, size=num_points, replace=True)
    parents = nd_points[parents]
    coefs = rng.dirichlet(np.ones(dims), size=num_points).reshape((num_points, 1, dims))
    points = np.matmul(coefs, parents).squeeze()
    if decimals is not None:
        points = np.around(points, decimals)
    return points


def sample_dominated_points(nd_points, num_dominated, sparse=True, decimals=None, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    dominators = rng.choice(nd_points, size=num_dominated, replace=True)
    diffs = rng.uniform(low=0, high=dominators)

    if sparse:
        dim = nd_points.shape[1]
        active = rng.choice([True, False], size=diffs.shape)
        for row in active:
            if np.sum(row) == 0:
                row[rng.integers(dim)] = True
        diffs = diffs * active

    d_points = dominators - diffs

    if decimals is not None:
        d_points = np.around(d_points, decimals)
    return d_points


class TestPruning(unittest.TestCase):
    test_seed = 0

    def make_set(self, points):
        return {tuple(vec) for vec in points}

    def test_small_pf(self):
        num_nd_points = 100
        num_d_points = 500
        dims = 2
        decimals = None
        min_val = 0
        max_val = 10
        rng = np.random.default_rng(self.test_seed)
        nd_points, d_points = generate_known_front(
            num_nd_points, num_d_points, dims=dims, decimals=decimals, min_val=min_val, max_val=max_val, rng=rng
        )
        computed_points = filter_pareto_dominated(np.vstack((nd_points, d_points)))
        self.assertEqual(self.make_set(nd_points), self.make_set(computed_points))

    def test_small_ch(self):
        num_nd_points = 100
        num_d_points = 500
        dims = 2
        decimals = None
        min_val = 0
        max_val = 10
        rng = np.random.default_rng(self.test_seed)
        nd_points, d_points = generate_known_front(
            num_nd_points, num_d_points, convex=True, dims=dims, decimals=decimals, min_val=min_val, max_val=max_val, rng=rng
        )
        computed_points = filter_convex_dominated(np.vstack((nd_points, d_points)))
        self.assertEqual(self.make_set(nd_points), self.make_set(computed_points))

    def test_large_pf(self):
        num_nd_points = 1000
        num_d_points = 5000
        dims = 4
        decimals = None
        min_val = 0
        max_val = 10
        rng = np.random.default_rng(self.test_seed)
        nd_points, d_points = generate_known_front(
            num_nd_points, num_d_points, dims=dims, decimals=decimals, min_val=min_val, max_val=max_val, rng=rng
        )
        computed_points = filter_pareto_dominated(np.vstack((nd_points, d_points)))
        self.assertEqual(self.make_set(nd_points), self.make_set(computed_points))

    def test_large_ch(self):
        num_nd_points = 1000
        num_d_points = 5000
        dims = 4
        decimals = None
        min_val = 0
        max_val = 10
        rng = np.random.default_rng(self.test_seed)
        nd_points, d_points = generate_known_front(
            num_nd_points, num_d_points, convex=True, dims=dims, decimals=decimals, min_val=min_val, max_val=max_val, rng=rng
        )
        computed_points = filter_convex_dominated(np.vstack((nd_points, d_points)))
        self.assertEqual(self.make_set(nd_points), self.make_set(computed_points))


if __name__ == "__main__":
    unittest.main()

"""Tests for Pareto front plotting utilities."""

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from morl_baselines.common.plots import plot_pareto_front


class TestPlotParetoFront:
    def test_2d_front_returns_figure(self):
        front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        fig = plot_pareto_front(front)
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.name == "rectilinear"
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_2d_front_with_ref_point_and_labels(self):
        front = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]
        fig = plot_pareto_front(
            front,
            ref_point=np.array([0.0, 0.0]),
            labels=["Return A", "Return B"],
        )
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Return A"
        assert ax.get_ylabel() == "Return B"
        # scatter for reference point
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_3d_front_returns_figure(self):
        front = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
            ]
        )
        fig = plot_pareto_front(front, ref_point=np.array([-0.1, -0.1, -0.1]))
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert isinstance(ax, Axes3D)
        plt.close(fig)

    def test_list_of_arrays(self):
        front = [np.array([0.2, 0.8]), np.array([0.4, 0.6]), np.array([0.9, 0.1])]
        fig = plot_pareto_front(front)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_unsupported_dimension_raises(self):
        front = np.ones((5, 4))
        with pytest.raises(ValueError, match="2D or 3D"):
            plot_pareto_front(front)

    def test_label_length_mismatch_raises(self):
        front = np.array([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="labels length"):
            plot_pareto_front(front, labels=["only-one"])


if __name__ == "__main__":
    pytest.main([__file__])

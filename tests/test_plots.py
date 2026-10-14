"""Tests for Pareto front plotting utilities."""

import matplotlib


matplotlib.use("Agg")

from unittest.mock import MagicMock, patch

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

    def test_2d_unsorted_front_is_sorted_by_first_objective(self):
        front = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        fig = plot_pareto_front(front)
        ax = fig.axes[0]
        line = ax.lines[0]
        xdata = np.asarray(line.get_xdata())
        assert np.all(np.diff(xdata) >= 0)
        np.testing.assert_allclose(xdata, [0.0, 0.5, 1.0])
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

    def test_3d_ref_point_and_axis_labels(self):
        front = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        fig = plot_pareto_front(
            front,
            ref_point=np.array([0.0, 0.0, 0.0]),
            labels=["Obj X", "Obj Y", "Obj Z"],
        )
        ax = fig.axes[0]
        assert isinstance(ax, Axes3D)
        assert ax.get_xlabel() == "Obj X"
        assert ax.get_ylabel() == "Obj Y"
        assert ax.get_zlabel() == "Obj Z"
        assert len(ax.collections) >= 2  # front + ref point
        plt.close(fig)

    def test_list_of_arrays(self):
        front = [np.array([0.2, 0.8]), np.array([0.4, 0.6]), np.array([0.9, 0.1])]
        fig = plot_pareto_front(front)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_single_point_1d_accepted(self):
        fig = plot_pareto_front(np.array([0.3, 0.7]))
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_empty_2d_front(self):
        fig = plot_pareto_front(np.empty((0, 2)))
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_empty_3d_front(self):
        fig = plot_pareto_front(np.empty((0, 3)))
        assert isinstance(fig, Figure)
        assert isinstance(fig.axes[0], Axes3D)
        plt.close(fig)

    def test_unsupported_dimension_raises(self):
        front = np.ones((5, 4))
        with pytest.raises(ValueError, match="2D or 3D"):
            plot_pareto_front(front)

    def test_bad_rank_raises(self):
        front = np.ones((2, 2, 2))
        with pytest.raises(ValueError, match="2D array-like"):
            plot_pareto_front(front)

    def test_label_length_mismatch_raises(self):
        front = np.array([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="labels length"):
            plot_pareto_front(front, labels=["only-one"])

    def test_incompatible_ref_point_2d_raises_before_figure(self):
        front = np.array([[0.0, 1.0], [1.0, 0.0]])
        n_before = len(plt.get_fignums())
        with pytest.raises(ValueError, match="ref_point must have shape"):
            plot_pareto_front(front, ref_point=np.array([0.0, 0.0, 0.0]))
        assert len(plt.get_fignums()) == n_before

    def test_incompatible_ref_point_3d_raises(self):
        front = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with pytest.raises(ValueError, match="ref_point must have shape"):
            plot_pareto_front(front, ref_point=np.array([0.0, 0.0]))


class TestLogParetoFrontFigureClosed:
    """Mocked checks that evaluation closes the figure even if wandb fails."""

    @staticmethod
    def _import_evaluation_with_stubs():
        """Import evaluation without requiring torch/pymoo/scipy at runtime."""
        import sys
        import types

        stubs = {}

        def ensure(name, attrs=None):
            if name in sys.modules and not isinstance(sys.modules[name], MagicMock):
                # Keep real modules (numpy, matplotlib, wandb, etc.)
                if name in ("numpy", "matplotlib", "matplotlib.pyplot", "wandb"):
                    return sys.modules[name]
            if name not in sys.modules or isinstance(sys.modules.get(name), MagicMock):
                mod = types.ModuleType(name)
                if attrs:
                    for k, v in attrs.items():
                        setattr(mod, k, v)
                stubs[name] = mod
                sys.modules[name] = mod
            return sys.modules[name]

        # Heavy / optional deps used only at import time
        torch = ensure("torch")
        torch.manual_seed = MagicMock()
        torch.cuda = MagicMock()
        torch.backends = MagicMock()

        ensure("pymoo")
        ensure("pymoo.util")
        ensure("pymoo.util.ref_dirs", {"get_reference_directions": MagicMock()})
        ensure("pymoo.indicators")
        ensure("pymoo.indicators.hv", {"HV": MagicMock()})
        ensure("pymoo.indicators.igd", {"IGD": MagicMock()})
        ensure("scipy")
        ensure("scipy.spatial", {"ConvexHull": MagicMock()})

        # Clear cached evaluation / sibling modules so stubs apply
        for key in list(sys.modules):
            if key.startswith("morl_baselines.common.evaluation") or key in (
                "morl_baselines.common.performance_indicators",
                "morl_baselines.common.pareto",
                "morl_baselines.common.weights",
            ):
                del sys.modules[key]

        from morl_baselines.common import evaluation

        return evaluation

    def test_figure_closed_on_success(self):
        evaluation = self._import_evaluation_with_stubs()

        front = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        ref = np.array([0.0, 0.0])
        with (
            patch.object(evaluation, "wandb") as mock_wandb,
            patch.object(evaluation, "plt") as mock_plt,
            patch.object(evaluation, "plot_pareto_front") as mock_plot,
            patch.object(evaluation, "hypervolume", return_value=1.0),
            patch.object(evaluation, "expected_utility", return_value=1.0),
            patch.object(evaluation, "cardinality", return_value=2),
            patch.object(evaluation, "filter_pareto_dominated", side_effect=lambda x: x),
            patch.object(evaluation, "equally_spaced_weights", return_value=np.eye(2)),
        ):
            fig = MagicMock(name="fig")
            mock_plot.return_value = fig
            mock_wandb.Image.return_value = "img"
            mock_wandb.Table.return_value = MagicMock()

            evaluation.log_all_multi_policy_metrics(
                current_front=front,
                hv_ref_point=ref,
                reward_dim=2,
                global_step=0,
                n_sample_weights=2,
            )

            mock_wandb.Image.assert_called_once_with(fig)
            mock_plt.close.assert_called_once_with(fig)

    def test_figure_closed_on_image_exception(self):
        evaluation = self._import_evaluation_with_stubs()

        front = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        ref = np.array([0.0, 0.0])
        with (
            patch.object(evaluation, "wandb") as mock_wandb,
            patch.object(evaluation, "plt") as mock_plt,
            patch.object(evaluation, "plot_pareto_front") as mock_plot,
            patch.object(evaluation, "hypervolume", return_value=1.0),
            patch.object(evaluation, "expected_utility", return_value=1.0),
            patch.object(evaluation, "cardinality", return_value=2),
            patch.object(evaluation, "filter_pareto_dominated", side_effect=lambda x: x),
            patch.object(evaluation, "equally_spaced_weights", return_value=np.eye(2)),
        ):
            fig = MagicMock(name="fig")
            mock_plot.return_value = fig
            mock_wandb.Image.side_effect = RuntimeError("wandb image failed")
            mock_wandb.Table.return_value = MagicMock()

            with pytest.raises(RuntimeError, match="wandb image failed"):
                evaluation.log_all_multi_policy_metrics(
                    current_front=front,
                    hv_ref_point=ref,
                    reward_dim=2,
                    global_step=0,
                    n_sample_weights=2,
                )

            mock_plt.close.assert_called_once_with(fig)

    def test_figure_closed_on_log_exception(self):
        evaluation = self._import_evaluation_with_stubs()

        front = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        ref = np.array([0.0, 0.0])
        with (
            patch.object(evaluation, "wandb") as mock_wandb,
            patch.object(evaluation, "plt") as mock_plt,
            patch.object(evaluation, "plot_pareto_front") as mock_plot,
            patch.object(evaluation, "hypervolume", return_value=1.0),
            patch.object(evaluation, "expected_utility", return_value=1.0),
            patch.object(evaluation, "cardinality", return_value=2),
            patch.object(evaluation, "filter_pareto_dominated", side_effect=lambda x: x),
            patch.object(evaluation, "equally_spaced_weights", return_value=np.eye(2)),
        ):
            fig = MagicMock(name="fig")
            mock_plot.return_value = fig
            mock_wandb.Image.return_value = "img"
            mock_wandb.Table.return_value = MagicMock()

            def log_side_effect(data, commit=True):
                if isinstance(data, dict) and "eval/pareto_front" in data:
                    raise RuntimeError("wandb log failed")

            mock_wandb.log.side_effect = log_side_effect

            with pytest.raises(RuntimeError, match="wandb log failed"):
                evaluation.log_all_multi_policy_metrics(
                    current_front=front,
                    hv_ref_point=ref,
                    reward_dim=2,
                    global_step=0,
                    n_sample_weights=2,
                )

            mock_plt.close.assert_called_once_with(fig)



if __name__ == "__main__":
    pytest.main([__file__])

"""Plotting utilities for MORL baselines."""

from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_pareto_front(
    front: Union[np.ndarray, List[np.ndarray], Sequence[Sequence[float]]],
    ref_point: Optional[np.ndarray] = None,
    labels: Optional[Sequence[str]] = None,
) -> Figure:
    """Plot a 2D or 3D Pareto front.

    For 2D fronts, points are sorted by the first objective and connected with a line.
    For 3D fronts, points are shown as a scatter plot with a 3D projection.

    Args:
        front: Pareto front points of shape (n_points, n_objectives) with 2 or 3 objectives.
        ref_point: Optional reference point to mark on the plot (e.g. hypervolume reference).
        labels: Optional axis labels for each objective. Defaults to ``Objective i``.

    Returns:
        matplotlib Figure containing the plot.

    Raises:
        ValueError: If the front does not have 2 or 3 objectives.
    """
    points = np.asarray(front, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2:
        raise ValueError(f"front must be 2D array-like, got shape {points.shape}")

    n_obj = points.shape[1]
    if n_obj not in (2, 3):
        raise ValueError(f"plot_pareto_front only supports 2D or 3D fronts, got {n_obj} objectives")

    if labels is None:
        labels = [f"Objective {i + 1}" for i in range(n_obj)]
    elif len(labels) != n_obj:
        raise ValueError(f"labels length ({len(labels)}) must match number of objectives ({n_obj})")

    fig = plt.figure()
    if n_obj == 2:
        ax = fig.add_subplot(111)
        order = np.argsort(points[:, 0])
        sorted_pts = points[order]
        ax.plot(sorted_pts[:, 0], sorted_pts[:, 1], "-o", label="Pareto front")
        if ref_point is not None:
            ref_point = np.asarray(ref_point, dtype=np.float64)
            ax.scatter(
                ref_point[0],
                ref_point[1],
                marker="*",
                s=150,
                c="red",
                label="Reference point",
                zorder=5,
            )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.grid(alpha=0.25)
        ax.legend()
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label="Pareto front")
        if ref_point is not None:
            ref_point = np.asarray(ref_point, dtype=np.float64)
            ax.scatter(
                ref_point[0],
                ref_point[1],
                ref_point[2],
                marker="*",
                s=150,
                c="red",
                label="Reference point",
            )
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.legend()

    ax.set_title("Pareto Front")
    fig.tight_layout()
    return fig

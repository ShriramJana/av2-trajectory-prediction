"""Scene plots in the focal frame: lanes, histories, ground truth, predicted modes."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_scene(
    bundle: dict[str, np.ndarray],
    traj: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    ax: Axes | None = None,
    radius: float = 60.0,
) -> Axes:
    """Draw one scenario. `bundle` is an npz bundle (unpadded arrays);
    `traj` (K, 60, 2) and `probs` (K,) are optional model outputs.

    Lanes grey, other agents light blue, focal history black, GT future green,
    predicted modes red with opacity growing with probability.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    for lane in bundle["lanes"][bundle["lane_mask"]]:
        ax.plot(lane[:, 0], lane[:, 1], color="0.78", lw=1, zorder=1)

    for hist, mask in zip(bundle["agent_hist"][1:], bundle["agent_mask"][1:]):
        if mask.any():
            ax.plot(hist[mask, 0], hist[mask, 1], color="#8ec1e8", lw=1.2, zorder=2)
            ax.plot(*hist[mask][-1], "o", color="#8ec1e8", ms=3, zorder=2)

    if traj is not None:
        probs = np.full(len(traj), 1.0 / len(traj)) if probs is None else probs
        for mode, p in sorted(zip(traj, probs), key=lambda tp: tp[1]):
            alpha = 0.25 + 0.75 * p / probs.max()
            ax.plot(mode[:, 0], mode[:, 1], color="#d62728", lw=1.8, alpha=alpha, zorder=4)
            ax.plot(*mode[-1], "o", color="#d62728", ms=4, alpha=alpha, zorder=4)
            if len(traj) > 1:
                ax.annotate(f"{p:.2f}", mode[-1], fontsize=7, color="#7f1d1d",
                            xytext=(3, 3), textcoords="offset points", zorder=6)

    focal = bundle["agent_hist"][0]
    ax.plot(focal[:, 0], focal[:, 1], color="black", lw=2.2, zorder=5, label="history")
    ax.plot(bundle["future"][:, 0], bundle["future"][:, 1], color="#2ca02c", lw=2.2,
            zorder=3, label="ground truth")
    ax.plot(*bundle["future"][-1], "*", color="#2ca02c", ms=11, zorder=5)

    # center the view between the origin and the GT endpoint so both stay visible
    cx, cy = bundle["future"][-1] / 2
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

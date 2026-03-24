"""Visualisation utilities for 2D point cloud data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from flows.data import MoonsDataset, MoonsSplits

CLASS_COLOURS = {0: "#1f77b4", 1: "#ff7f0e"}


def scatter(
    ds: MoonsDataset,
    ax: Axes | None = None,
    alpha: float = 0.6,
    s: float = 12,
) -> Axes:
    """Scatter plot of a single split, coloured by class."""
    if ax is None:
        _, ax = plt.subplots()
    x = ds.x.numpy()
    labels = ds.labels.numpy()
    for c in np.unique(labels):
        mask = labels == c
        ax.scatter(x[mask, 0], x[mask, 1], s=s, alpha=alpha, label=f"class {c}", color=CLASS_COLOURS[c])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(ds.name)
    ax.legend(markerscale=2)
    ax.set_aspect("equal")
    return ax


def scatter_splits(splits: MoonsSplits, figsize: tuple[float, float] = (14, 4)) -> Figure:
    """Side-by-side scatter of train / val / test."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, ds in zip(axes, [splits.train, splits.val, splits.test]):
        scatter(ds, ax=ax)
    fig.tight_layout()
    return fig


def marginal_histograms(
    ds: MoonsDataset,
    bins: int = 40,
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """Per-class marginal histograms for x1 and x2."""
    x = ds.x.numpy()
    labels = ds.labels.numpy()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for i, name in enumerate(["$x_1$", "$x_2$"]):
        for c in np.unique(labels):
            axes[i].hist(
                x[labels == c, i],
                bins=bins,
                alpha=0.5,
                label=f"class {c}",
                color=CLASS_COLOURS[c],
                density=True,
            )
        axes[i].set_xlabel(name)
        axes[i].set_ylabel("density")
        axes[i].legend()
        axes[i].set_title(f"{ds.name} — {name} marginal")
    fig.tight_layout()
    return fig


def joint_kde(
    ds: MoonsDataset,
    gridsize: int = 80,
    figsize: tuple[float, float] = (6, 5),
) -> Figure:
    """Hexbin density estimate of the joint distribution."""
    x = ds.x.numpy()
    fig, ax = plt.subplots(figsize=figsize)
    hb = ax.hexbin(x[:, 0], x[:, 1], gridsize=gridsize, cmap="viridis", mincnt=1)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"{ds.name} — joint density")
    ax.set_aspect("equal")
    fig.colorbar(hb, ax=ax, label="count")
    fig.tight_layout()
    return fig


def pairplot(
    ds: MoonsDataset,
    figsize: tuple[float, float] = (7, 7),
    bins: int = 30,
) -> Figure:
    """2x2 pairplot: diagonals are marginal histograms, off-diagonals are scatter."""
    x = ds.x.numpy()
    labels = ds.labels.numpy()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    dim_names = ["$x_1$", "$x_2$"]

    for i in range(2):
        # diagonal: histogram
        for c in np.unique(labels):
            axes[i][i].hist(
                x[labels == c, i], bins=bins, alpha=0.5,
                label=f"class {c}", color=CLASS_COLOURS[c], density=True,
            )
        axes[i][i].set_xlabel(dim_names[i])
        axes[i][i].legend(fontsize=8)

        # off-diagonal: scatter
        j = 1 - i
        for c in np.unique(labels):
            mask = labels == c
            axes[i][j].scatter(
                x[mask, j], x[mask, i], s=8, alpha=0.5,
                color=CLASS_COLOURS[c], label=f"class {c}",
            )
        axes[i][j].set_xlabel(dim_names[j])
        axes[i][j].set_ylabel(dim_names[i])

    fig.suptitle(f"{ds.name} — pairplot", y=1.01)
    fig.tight_layout()
    return fig


def knn_distances(
    ds: MoonsDataset,
    k: int = 5,
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """Histogram of k-NN distances per class, revealing local density variation."""
    from scipy.spatial import KDTree

    x = ds.x.numpy()
    labels = ds.labels.numpy()
    tree = KDTree(x)
    dists, _ = tree.query(x, k=k + 1)  # +1 because nearest neighbour is self
    knn_dists = dists[:, 1:].mean(axis=1)  # mean distance to k nearest neighbours

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Per-class histogram
    for c in np.unique(labels):
        mask = labels == c
        axes[0].hist(knn_dists[mask], bins=30, alpha=0.5, label=f"class {c}",
                     color=CLASS_COLOURS[c], density=True)
    axes[0].set_xlabel(f"Mean distance to {k} nearest neighbours")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"{ds.name} — {k}-NN distance distribution")
    axes[0].legend()

    # Scatter coloured by local density (inverse knn distance)
    sc = axes[1].scatter(x[:, 0], x[:, 1], c=knn_dists, s=12, cmap="viridis_r", alpha=0.7)
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")
    axes[1].set_title(f"{ds.name} — local density (darker = denser)")
    axes[1].set_aspect("equal")
    fig.colorbar(sc, ax=axes[1], label=f"mean {k}-NN dist")

    fig.tight_layout()
    return fig


def covariance_ellipses(
    ds: MoonsDataset,
    n_std: float = 2.0,
    figsize: tuple[float, float] = (7, 6),
) -> Figure:
    """Scatter with per-class covariance ellipses overlaid."""
    x = ds.x.numpy()
    labels = ds.labels.numpy()
    fig, ax = plt.subplots(figsize=figsize)

    for c in np.unique(labels):
        mask = labels == c
        xc = x[mask]
        mean = xc.mean(axis=0)
        cov = np.cov(xc, rowvar=False)

        ax.scatter(xc[:, 0], xc[:, 1], s=10, alpha=0.4, color=CLASS_COLOURS[c], label=f"class {c}")

        # Eigen-decomposition for ellipse orientation and axes
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        for ns, alpha in [(1, 0.3), (n_std, 0.15)]:
            width, height = 2 * ns * np.sqrt(eigenvalues)
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                              facecolor=CLASS_COLOURS[c], alpha=alpha, edgecolor=CLASS_COLOURS[c],
                              linewidth=1.5, linestyle="--" if ns > 1 else "-")
            ax.add_patch(ellipse)
        ax.plot(*mean, "x", color=CLASS_COLOURS[c], markersize=10, markeredgewidth=2)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"{ds.name} — per-class covariance ellipses ({n_std}σ)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def radial_from_centroids(
    ds: MoonsDataset,
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """Histogram of distances from each point to its class centroid."""
    x = ds.x.numpy()
    labels = ds.labels.numpy()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for c in np.unique(labels):
        mask = labels == c
        xc = x[mask]
        centroid = xc.mean(axis=0)
        dists = np.linalg.norm(xc - centroid, axis=1)
        axes[0].hist(dists, bins=25, alpha=0.5, label=f"class {c}",
                     color=CLASS_COLOURS[c], density=True)

    axes[0].set_xlabel("Distance from class centroid")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"{ds.name} — radial distance distribution")
    axes[0].legend()

    # Polar-style: angle from centroid vs distance
    for c in np.unique(labels):
        mask = labels == c
        xc = x[mask]
        centroid = xc.mean(axis=0)
        delta = xc - centroid
        angles = np.arctan2(delta[:, 1], delta[:, 0])
        dists = np.linalg.norm(delta, axis=1)
        axes[1].scatter(angles, dists, s=8, alpha=0.5, color=CLASS_COLOURS[c], label=f"class {c}")

    axes[1].set_xlabel("Angle from centroid (rad)")
    axes[1].set_ylabel("Distance from centroid")
    axes[1].set_title(f"{ds.name} — angular structure")
    axes[1].legend()

    fig.tight_layout()
    return fig


def qq_splits(
    splits: MoonsSplits,
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """QQ plots comparing train vs val and train vs test for each feature."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    dim_names = ["$x_1$", "$x_2$"]

    for i, name in enumerate(dim_names):
        train_sorted = np.sort(splits.train.x[:, i].numpy())
        for other, style in [(splits.val, "o"), (splits.test, "s")]:
            other_sorted = np.sort(other.x[:, i].numpy())
            # Interpolate train quantiles to match other's length
            train_quantiles = np.interp(
                np.linspace(0, 1, len(other_sorted)),
                np.linspace(0, 1, len(train_sorted)),
                train_sorted,
            )
            axes[i].plot(train_quantiles, other_sorted, style, markersize=3,
                         alpha=0.6, label=f"train vs {other.name}")
        # Reference line
        lims = [axes[i].get_xlim(), axes[i].get_ylim()]
        lo = min(lims[0][0], lims[1][0])
        hi = max(lims[0][1], lims[1][1])
        axes[i].plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
        axes[i].set_xlabel(f"Train {name} quantiles")
        axes[i].set_ylabel(f"Other {name} quantiles")
        axes[i].set_title(f"QQ plot — {name}")
        axes[i].legend()
        axes[i].set_aspect("equal")

    fig.tight_layout()
    return fig


def base_density_overlay(
    ds: MoonsDataset,
    figsize: tuple[float, float] = (7, 6),
) -> Figure:
    """Scatter of data with standard normal density contours overlaid.

    Shows the gap between the data distribution and the base N(0,I)
    that the flow must bridge.
    """
    x = ds.x.numpy()
    labels = ds.labels.numpy()
    fig, ax = plt.subplots(figsize=figsize)

    # Data scatter
    for c in np.unique(labels):
        mask = labels == c
        ax.scatter(x[mask, 0], x[mask, 1], s=10, alpha=0.5,
                   color=CLASS_COLOURS[c], label=f"class {c}")

    # Standard normal contours
    pad = 0.5
    lo = min(x[:, 0].min(), x[:, 1].min(), -3) - pad
    hi = max(x[:, 0].max(), x[:, 1].max(), 3) + pad
    grid = np.linspace(lo, hi, 200)
    X, Y = np.meshgrid(grid, grid)
    pos = np.stack([X.ravel(), Y.ravel()], axis=-1)
    Z = multivariate_normal(mean=[0, 0], cov=np.eye(2)).pdf(pos).reshape(X.shape)

    contour = ax.contour(X, Y, Z, levels=6, cmap="Greys", alpha=0.6, linewidths=1)
    ax.clabel(contour, inline=True, fontsize=7, fmt="%.3f")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"{ds.name} — data vs base density $\\mathcal{{N}}(0, I)$")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def class_balance_bar(
    *datasets: MoonsDataset,
    figsize: tuple[float, float] = (8, 3),
) -> Figure:
    """Grouped bar chart of class counts across splits."""
    names = [ds.name for ds in datasets]
    classes = sorted(set(int(c) for ds in datasets for c in ds.labels.unique().tolist()))
    counts = {c: [int((ds.labels == c).sum()) for ds in datasets] for c in classes}

    x_pos = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    for i, c in enumerate(classes):
        offset = (i - (len(classes) - 1) / 2) * width
        bars = ax.bar(x_pos + offset, counts[c], width, label=f"class {c}", color=CLASS_COLOURS[c])
        for bar, val in zip(bars, counts[c]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(val),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_ylabel("count")
    ax.set_title("Class balance across splits")
    ax.legend()
    fig.tight_layout()
    return fig


def figure1c(
    recon_errors: torch.Tensor,
    max_abs_error: float,
    jacobian: torch.Tensor,
    logdet_abs_error: float,
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """Two-panel diagnostic figure for Q1(c).

    Panel 1: per-sample reconstruction error.
    Panel 2: numerical Jacobian heatmap.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: reconstruction error per sample
    axes[0].plot(recon_errors.numpy(), ".", markersize=2)
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Max absolute error")
    axes[0].set_title(f"Invertibility (max = {max_abs_error:.2e})")
    axes[0].axhline(max_abs_error, color="r", linestyle="--", linewidth=0.8)

    # Panel 2: Jacobian heatmap
    J_np = jacobian.numpy()
    im = axes[1].imshow(J_np, aspect="equal")
    for i in range(J_np.shape[0]):
        for j in range(J_np.shape[1]):
            axes[1].text(
                j, i, f"{J_np[i, j]:.4f}", ha="center", va="center",
                color="black", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )
    axes[1].set_title(
        f"Numerical Jacobian of $f^{{-1}}$\nlog-det error = {logdet_abs_error:.2e}"
    )
    axes[1].set_xlabel("Input dim")
    axes[1].set_ylabel("Output dim")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    fig.colorbar(im, ax=axes[1])

    fig.tight_layout()
    return fig


def figure2a(
    train_losses: list[float],
    val_losses: list[float],
    figsize: tuple[float, float] = (6, 4),
) -> Figure:
    """Training curve figure for Q2(a): tiny-subset training.

    Plots train and validation NLL over steps, with a vertical line
    at the step where validation NLL is minimised.
    """
    min_loss = min(val_losses)
    min_idx = int(np.argmin(val_losses))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(
        x=min_idx, color="red", linestyle="--",
        label=f"Min val NLL = {min_loss:.3f} (step {min_idx})",
    )
    ax.plot(train_losses, label="Train NLL", alpha=0.7)
    ax.plot(val_losses, label="Val NLL", alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("NLL")
    ax.set_title("Q2(a): Tiny subset (128 samples) training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def figure2c(
    ablation_results: list[dict],
    best_label: str,
    naive_losses: list[float],
    naive_val_losses: list[float],
    figsize: tuple[float, float] = (15, 4),
) -> Figure:
    """Three-panel regularisation ablation figure for Q2(c).

    Parameters
    ----------
    ablation_results : list[dict]
        Each dict must have keys ``label`` (str) and ``result`` (dict
        with ``train_losses``, ``val_losses``, and optionally
        ``best_val_step``).
    best_label : str
        The ``label`` of the best ablation config (drawn bold).
    naive_losses, naive_val_losses : list[float]
        Baseline (no-regularisation) curves shown faintly behind each panel.
    """
    n_panels = len(ablation_results)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True)

    for ax, ab in zip(axes, ablation_results):
        r = ab["result"]
        # Naive baseline as faint background
        ax.plot(naive_losses, color="grey", alpha=0.2, label="Naive train")
        ax.plot(naive_val_losses, color="grey", alpha=0.2, linestyle="--", label="Naive val")
        # Ablation curves
        ax.plot(r["train_losses"], color="C0", label="Train")
        ax.plot(r["val_losses"], color="C1", label="Val")
        if "best_val_step" in r:
            ax.axvline(
                x=r["best_val_step"], color="red", linestyle=":", alpha=0.7,
                label=f"Best val (step {r['best_val_step'] + 1})",
            )
        ax.set_title(ab["label"], fontweight="bold" if ab["label"] == best_label else "normal")
        ax.set_xlabel("Step")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("NLL")
    fig.suptitle("Q2(c): Regularisation ablation (naive baseline in grey)")
    fig.tight_layout()
    return fig


def samples_vs_data(
    generated: torch.Tensor,
    train_data: torch.Tensor,
    train_labels: torch.Tensor | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> Figure:
    """Side-by-side comparison of generated flow samples and training data.

    Left panel: training data (coloured by class if labels provided).
    Right panel: samples drawn from the flow.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: training data
    x = train_data.numpy()
    if train_labels is not None:
        labels = train_labels.numpy()
        for c in np.unique(labels):
            mask = labels == c
            axes[0].scatter(
                x[mask, 0], x[mask, 1], s=8, alpha=0.5,
                color=CLASS_COLOURS[c], label=f"class {c}",
            )
        axes[0].legend(markerscale=2)
    else:
        axes[0].scatter(x[:, 0], x[:, 1], s=8, alpha=0.5, color="C0")
    axes[0].set_title("Training data")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].set_aspect("equal")

    # Right: generated samples
    g = generated.numpy()
    axes[1].scatter(g[:, 0], g[:, 1], s=8, alpha=0.5, color="C2")
    axes[1].set_title(f"Generated samples (n={len(g)})")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")
    axes[1].set_aspect("equal")

    fig.tight_layout()
    return fig


def figure3b(
    samples: dict[int | float, torch.Tensor],
    train_data: torch.Tensor,
    train_labels: torch.Tensor | None = None,
    figsize: tuple[float, float] = (15, 10),
) -> Figure:
    """2x3 panel figure for Q3(b): flow surgery samples.

    Top-left panel shows the original training distribution for reference.
    Remaining panels show generated samples for each alpha value, with the
    training data underlaid in grey.

    Parameters
    ----------
    samples : dict
        Mapping from alpha value to a ``[N, 2]`` tensor of generated samples.
    train_data : Tensor[N, 2]
        Training data points, underlaid on every surgery panel.
    train_labels : Tensor[N] or None
        Class labels for colouring the training-data panel.
    """
    alphas = sorted(samples.keys())
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes_flat = axes.flatten()

    # First panel: original training data
    ax0 = axes_flat[0]
    x = train_data.numpy()
    if train_labels is not None:
        labels = train_labels.numpy()
        for c in np.unique(labels):
            mask = labels == c
            ax0.scatter(
                x[mask, 0], x[mask, 1], s=8, alpha=0.5,
                color=CLASS_COLOURS[c], label=f"class {c}",
            )
        ax0.legend(markerscale=2, fontsize=7)
    else:
        ax0.scatter(x[:, 0], x[:, 1], s=8, alpha=0.5, color="C0")
    ax0.set_title("Training data")
    ax0.set_xlabel("$x_1$")
    ax0.set_ylabel("$x_2$")
    ax0.set_aspect("equal")

    # Remaining panels: surgery samples with training data underlay
    for ax, a in zip(axes_flat[1:], alphas):
        ax.scatter(x[:, 0], x[:, 1], s=4, alpha=0.15, color="grey", zorder=1)
        s = samples[a].numpy()
        ax.scatter(s[:, 0], s[:, 1], s=5, alpha=0.4, zorder=2)
        ax.set_title(f"$\\alpha = {a}$")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")

    # Hide any unused axes
    for ax in axes_flat[1 + len(alphas):]:
        ax.set_visible(False)

    fig.suptitle("Generated samples from $p_\\alpha(x)$")
    fig.tight_layout()
    return fig

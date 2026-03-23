"""Q1(c) correctness checks — invertibility and log-det verification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from flows.model import Flow


def check_invertibility(flow: Flow, x: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Round-trip x -> z -> x_hat and return (max_abs_error, per_sample_errors).

    Parameters
    ----------
    flow : Flow
        A flow model (can be untrained).
    x : torch.Tensor
        Data tensor of shape (N, D).

    Returns
    -------
    max_abs_error : float
        Peak reconstruction error across all samples and dimensions.
    recon_errors : torch.Tensor
        Per-sample max absolute error, shape (N,).
    """
    flow_d = flow.double()
    flow_d.eval()
    x_d = x.double()
    with torch.no_grad():
        # encode the entire training set
        z, _ = flow_d.inverse(x_d)
        # decode the data
        x_hat, _ = flow_d.forward(z)
        # extract the peak inversion error
        recon_errors = (x_d - x_hat).abs().max(dim=-1).values
    return recon_errors.max().item(), recon_errors.float()


def check_logdet(flow: Flow, x0: torch.Tensor, eps: float = 1e-4) -> dict:
    """Compare analytic log|det J| to a central-difference estimate.

    Parameters
    ----------
    flow : Flow
        A flow model.
    x0 : torch.Tensor
        A single data point, shape (1, D).
    eps : float
        Step size for finite differences.

    Returns
    -------
    dict with keys: analytic_logdet, numerical_logdet, abs_error, jacobian.
    """
    flow_d = flow.double()
    flow_d.eval()
    x0_d = x0.double()
    dim = x0_d.shape[-1]

    # Analytic log-det from our inverse
    with torch.no_grad():
        _, analytic_log_det = flow_d.inverse(x0_d)
        analytic_log_det = analytic_log_det.item()

    # Numerical Jacobian by central differences
    J = torch.zeros(dim, dim, dtype=torch.float64)
    # iterate along elements of x0 (x0_1 and x0_2)
    for j in range(dim):
        e_j = torch.zeros(1, dim, dtype=torch.float64)
        e_j[0, j] = eps
        with torch.no_grad():
            # calculate change in both z1 and z2 relative to x0_j
            z_plus, _ = flow_d.inverse(x0_d + e_j)
            z_minus, _ = flow_d.inverse(x0_d - e_j)
        # squeeze removes the batch dimension
        J[:, j] = (z_plus - z_minus).squeeze() / (2 * eps)

    numerical_log_det = torch.log(torch.abs(torch.det(J))).item()
    abs_error = abs(analytic_log_det - numerical_log_det)

    return {
        "analytic_logdet": analytic_log_det,
        "numerical_logdet": numerical_log_det,
        "abs_error": abs_error,
        "jacobian": J,
    }


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
    im = axes[1].imshow(jacobian.numpy(), aspect="equal")
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

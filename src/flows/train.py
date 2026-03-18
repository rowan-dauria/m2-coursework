"""Training pipeline utilities for normalising flows."""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from flows.model import Flow


def evaluate_nll(model: Flow, data: torch.Tensor) -> float:
    """Compute the mean negative log-likelihood of *data* under *model*.

    The model is set to eval mode and gradients are disabled for the
    forward pass.  Returns a plain Python float.
    """
    model.eval()
    with torch.no_grad():
        nll = -model.log_prob(data).mean().item()
    return nll


def train_flow(
    model: Flow,
    x_train: torch.Tensor,
    x_val: torch.Tensor | None = None,
    *,
    n_steps: int = 5000,
    lr: float = 1e-3,
    use_cosine_schedule: bool = False,
    log_every: int = 500,
) -> dict[str, Any]:
    """Train a normalising flow by minimising the mean NLL.

    Parameters
    ----------
    model : nn.Module
        A flow model that exposes ``log_prob(x) -> Tensor[B]``.
    x_train : Tensor[N, D]
        Training data (used in full each step, no mini-batching).
    x_val : Tensor[M, D] or None
        Optional validation data.  When provided the validation NLL is
        recorded every step.
    n_steps : int
        Number of optimiser steps.
    lr : float
        Initial learning rate for Adam.
    use_cosine_schedule : bool
        If *True*, wrap the optimiser with a ``CosineAnnealingLR``
        scheduler (``T_max=n_steps``).
    log_every : int
        Print a progress line every this many steps.  Set to 0 to
        suppress printing.

    Returns
    -------
    dict with keys:
        ``train_losses``  – list[float] of per-step training NLL
        ``val_losses``    – list[float] of per-step validation NLL
                           (empty list when *x_val* is None)
        ``final_train_nll`` – float, final training NLL after the loop
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
        if use_cosine_schedule
        else None
    )

    train_losses: list[float] = []
    val_losses: list[float] = []

    for step in range(n_steps):
        # --- train step ---
        model.train()
        loss = -model.log_prob(x_train).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        train_nll = loss.item()
        train_losses.append(train_nll)

        # --- validation ---
        val_nll: float | None = None
        if x_val is not None:
            val_nll = evaluate_nll(model, x_val)
            val_losses.append(val_nll)

        # --- logging ---
        if log_every > 0 and (step + 1) % log_every == 0:
            msg = f"Step {step + 1}/{n_steps}, NLL: {train_nll:.4f}"
            if val_nll is not None:
                msg += f", Val NLL: {val_nll:.4f}"
            print(msg)

    # Final NLL evaluated cleanly (no grad, eval mode)
    final_train_nll = evaluate_nll(model, x_train)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_nll": final_train_nll,
    }


def save_checkpoint(
    model: Flow,
    config: dict[str, Any],
    seed: int,
    path: str = "checkpoints/flow_full.pt",
) -> None:
    """Save a flow checkpoint to *path*.

    Creates parent directories as needed.  The checkpoint is a dict
    with keys ``state_dict``, ``config``, and ``seed``.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "seed": seed,
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


def save_training_curves(
    curves: dict[str, list[float]],
    path: str = "logs/training_curves.json",
) -> None:
    """Write training-curve data to a JSON file.

    Creates parent directories as needed.

    Parameters
    ----------
    curves : dict[str, list[float]]
        Mapping of curve names to lists of per-step values.  Typical
        keys: ``tiny_loss``, ``full_loss``, ``full_val_loss``.
    path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(curves, f)
    print(f"Saved training curves to {path}")

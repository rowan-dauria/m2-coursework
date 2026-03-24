"""Training pipeline utilities for normalising flows."""

from __future__ import annotations

import copy
import itertools
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
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = 1.0,
    use_cosine_schedule: bool = False,
    early_stopping_patience: int | None = None,
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
    weight_decay : float
        L2 regularisation coefficient for Adam (default 0).
    grad_clip_norm : float or None
        If not None, clip gradients to this max norm each step.
        Important for flow stability.
    use_cosine_schedule : bool
        If *True*, wrap the optimiser with a ``CosineAnnealingLR``
        scheduler (``T_max=n_steps``).
    early_stopping_patience : int or None
        If not None (and *x_val* is provided), restore the best
        model weights when validation NLL has not improved for this
        many steps.
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
        ``best_val_step``   – int, step at which best val NLL was seen
                              (only present when early stopping is active)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
        if use_cosine_schedule
        else None
    )

    train_losses: list[float] = []
    val_losses: list[float] = []

    # Early stopping state
    use_early_stopping = (
        early_stopping_patience is not None and x_val is not None
    )
    best_val_nll = float("inf")
    best_val_step = 0
    best_state_dict: dict[str, Any] | None = None
    patience_counter = 0

    for step in range(n_steps):
        # --- train step ---
        model.train()
        loss = -model.log_prob(x_train).mean()
        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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

        # --- early stopping bookkeeping ---
        if use_early_stopping and val_nll is not None:
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_val_step = step
                best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if early_stopping_patience is not None \
                and patience_counter >= early_stopping_patience:
                    if log_every > 0:
                        print(
                            f"Early stopping at step {step + 1} "
                            f"(best val NLL {best_val_nll:.4f} at step {best_val_step + 1})"
                        )
                    break

        # --- logging ---
        if log_every > 0 and (step + 1) % log_every == 0:
            msg = f"Step {step + 1}/{n_steps}, NLL: {train_nll:.4f}"
            if val_nll is not None:
                msg += f", Val NLL: {val_nll:.4f}"
            print(msg)

    # Restore best weights if early stopping was used
    if use_early_stopping and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if log_every > 0:
            print(f"Restored best weights from step {best_val_step + 1}")

    # Final NLL evaluated cleanly (no grad, eval mode)
    final_train_nll = evaluate_nll(model, x_train)

    result: dict[str, Any] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_nll": final_train_nll,
    }
    if use_early_stopping:
        result["best_val_step"] = best_val_step
    return result


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


def update_results(
    updates: dict[str, Any],
    path: str = "results.json",
) -> dict[str, Any]:
    """Merge *updates* into a JSON results file (creating it if needed).

    Top-level keys in *updates* are merged into the existing dict; nested
    values are replaced wholesale.  Returns the full merged dict.
    """
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
    else:
        results = {}
    results.update(updates)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_ablation(
    configs: list[tuple[str, float, float]],
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    x_test: torch.Tensor,
    *,
    dim: int,
    hidden: int,
    n_layers: int,
    n_steps: int,
    lr: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Train one model per regularisation config and evaluate each.

    Parameters
    ----------
    configs : list of (label, clip_norm, weight_decay) tuples
    """
    results: list[dict[str, Any]] = []
    for label, clip_norm, wd in configs:
        torch.manual_seed(seed)
        model = Flow(dim=dim, hidden=hidden, n_layers=n_layers)
        result = train_flow(
            model, x_train, x_val,
            n_steps=n_steps,
            lr=lr,
            weight_decay=wd,
            grad_clip_norm=clip_norm,
            use_cosine_schedule=True,
            early_stopping_patience=1000,
            log_every=0,
        )
        val_nll = evaluate_nll(model, x_val)
        test_nll = evaluate_nll(model, x_test)
        steps_used = len(result["train_losses"])
        results.append({
            "label": label,
            "model": model,
            "result": result,
            "val_nll": val_nll,
            "test_nll": test_nll,
            "clip_norm": clip_norm,
            "weight_decay": wd,
        })
        print(f"  {label:20s}  val={val_nll:.4f}  steps={steps_used}")
    return results


def run_hp_scan(
    scan_lrs: list[float],
    scan_hiddens: list[int],
    scan_layers: list[int],
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    *,
    dim: int,
    n_steps: int,
    weight_decay: float,
    grad_clip_norm: float,
    seed: int,
    early_stopping_patience: int = 500,
) -> list[dict[str, Any]]:
    """Grid search over (lr, hidden, n_layers).

    Returns results sorted by validation NLL (best first).
    """
    results: list[dict[str, Any]] = []
    for lr, hidden, n_layers in itertools.product(scan_lrs, scan_hiddens, scan_layers):
        torch.manual_seed(seed)
        model = Flow(dim=dim, hidden=hidden, n_layers=n_layers)
        result = train_flow(
            model, x_train, x_val,
            n_steps=n_steps,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            use_cosine_schedule=True,
            early_stopping_patience=early_stopping_patience,
            log_every=0,
        )
        val_nll = evaluate_nll(model, x_val)
        steps_used = len(result["train_losses"])
        results.append({
            "lr": lr,
            "hidden": hidden,
            "n_layers": n_layers,
            "val_nll": val_nll,
            "train_nll": result["final_train_nll"],
            "steps": steps_used,
        })
        print(f"  lr={lr:.0e}  H={hidden:>3}  K={n_layers}  val={val_nll:.4f}  steps={steps_used}")
    results.sort(key=lambda r: r["val_nll"])
    return results


def print_scan_results(scan_results: list[dict[str, Any]], top_n: int = 10) -> None:
    """Print a ranked table of hyperparameter scan results."""
    print("\n" + "=" * 70)
    print(f"{'Rank':>4}  {'LR':>8}  {'H':>4}  {'K':>2}  {'Val NLL':>9}  {'Train NLL':>10}  {'Steps':>6}")
    print("-" * 70)
    for i, r in enumerate(scan_results[:top_n]):
        print(f"{i+1:>4}  {r['lr']:>8.0e}  {r['hidden']:>4}  {r['n_layers']:>2}  "
              f"{r['val_nll']:>9.4f}  {r['train_nll']:>10.4f}  {r['steps']:>6}")
    print("=" * 70)


def print_ablation_summary(
    ablation_results: list[dict[str, Any]],
    best: dict[str, Any],
    naive_train_nll: float | None = None,
    naive_val_nll: float | None = None,
    naive_steps: int = 10000,
) -> None:
    """Print a formatted summary table of ablation results."""
    print("\n" + "=" * 75)
    print(f"{'Config':20s}  {'Train NLL':>10}  {'Val NLL':>10}  {'Test NLL':>10}  {'Steps':>6}")
    print("-" * 75)
    if naive_train_nll is not None:
        naive_val_str = f"{naive_val_nll:>10.4f}" if naive_val_nll is not None else f"{'--':>10}"
        print(f"{'Naive (no reg.)':20s}  {naive_train_nll:>10.4f}  {naive_val_str}  {'--':>10}  {naive_steps:>6}")
    for ab in ablation_results:
        steps = len(ab["result"]["train_losses"])
        marker = "  <-- best" if ab is best else ""
        print(f"{ab['label']:20s}  {ab['result']['final_train_nll']:>10.4f}  "
              f"{ab['val_nll']:>10.4f}  {ab['test_nll']:>10.4f}  {steps:>6}{marker}")
    print("=" * 75)

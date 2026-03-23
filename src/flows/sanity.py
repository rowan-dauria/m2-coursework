"""Sanity checks for initial training results."""

from __future__ import annotations

import math

import torch

from flows.model import Flow
from flows.train import evaluate_nll, train_flow


def expected_initial_nll(x: torch.Tensor) -> float:
    """NLL of the data under a standard normal (the base density).

    A randomly initialised coupling flow with tanh-bounded scales starts
    close to the identity map, so its initial NLL should be near this value.
    """
    D = x.shape[1]
    log_pz = -0.5 * (D * math.log(2 * math.pi) + (x ** 2).sum(dim=-1))
    return -log_pz.mean().item()


def gaussian_baseline_nll(x_train: torch.Tensor, x_eval: torch.Tensor) -> float:
    """NLL of an input-independent Gaussian baseline fitted to the training data.

    Fits a full-covariance Gaussian to *x_train* and evaluates the mean NLL
    on *x_eval*.  Any trained model should comfortably beat this.
    """
    D = x_train.shape[1]
    mu = x_train.mean(dim=0)
    diff = x_train - mu
    cov = (diff.T @ diff) / (x_train.shape[0] - 1)

    # log p(x) = -0.5 * (D log(2pi) + log|det cov| + (x-mu)^T cov^{-1} (x-mu))
    cov_inv = torch.linalg.inv(cov)
    _, logdet = torch.linalg.slogdet(cov)

    diff_eval = x_eval - mu
    sq_mahalanobis = (diff_eval @ cov_inv * diff_eval).sum(dim=-1)
    log_px = -0.5 * (D * math.log(2 * math.pi) + logdet + sq_mahalanobis)
    return -log_px.mean().item()


def check_initial_loss(
    model: Flow,
    x: torch.Tensor,
    *,
    rtol: float = 0.5,
) -> dict:
    """Check that the untrained model's NLL is close to the standard-normal NLL.

    Returns a dict with the actual initial NLL, the expected value, and
    whether the check passed (within *rtol* relative tolerance).
    """
    initial_nll = evaluate_nll(model, x)
    expected = expected_initial_nll(x)
    rel_error = abs(initial_nll - expected) / abs(expected)
    passed = rel_error <= rtol

    return {
        "initial_nll": initial_nll,
        "expected_nll": expected,
        "rel_error": rel_error,
        "passed": passed,
    }


def check_beats_baseline(
    model: Flow,
    x_train: torch.Tensor,
    x_eval: torch.Tensor,
) -> dict:
    """Check that the trained model beats a fitted-Gaussian baseline on *x_eval*.

    Returns a dict with both NLLs, the margin, and whether the model wins.
    """
    model_nll = evaluate_nll(model, x_eval)
    baseline_nll = gaussian_baseline_nll(x_train, x_eval)
    margin = baseline_nll - model_nll
    passed = model_nll < baseline_nll

    return {
        "model_nll": model_nll,
        "baseline_nll": baseline_nll,
        "margin": margin,
        "passed": passed,
    }


def check_stable_start(
    model: Flow,
    x: torch.Tensor,
    *,
    n_steps: int = 50,
    lr: float = 1e-3,
) -> dict:
    """Check that the model trains stably for the first few steps.

    Trains for *n_steps* and verifies:
      1. No NaN/Inf losses appeared.
      2. The final loss is no worse than twice the initial loss (no explosion).
      3. The loss decreased overall.

    Returns a dict with diagnostics and a pass/fail flag.
    """
    result = train_flow(model, x, n_steps=n_steps, lr=lr, log_every=0)
    losses = result["train_losses"]

    has_nan = any(math.isnan(l) or math.isinf(l) for l in losses)
    initial = losses[0]
    final = losses[-1]
    no_explosion = final < 2 * initial
    decreased = final < initial

    return {
        "losses": losses,
        "initial_loss": initial,
        "final_loss": final,
        "has_nan": has_nan,
        "no_explosion": no_explosion,
        "decreased": decreased,
        "passed": (not has_nan) and no_explosion and decreased,
    }


def run_all_sanity_checks(
    x_train: torch.Tensor,
    x_eval: torch.Tensor,
    *,
    dim: int = 2,
    hidden: int = 128,
    n_layers: int = 8,
    seed: int = 42,
) -> dict:
    """Run all three sanity checks and print a summary.

    Creates a fresh model internally so the checks don't mutate your
    training model.
    """
    torch.manual_seed(seed)
    model = Flow(dim=dim, hidden=hidden, n_layers=n_layers)

    print("=" * 60)
    print("TRAINING SANITY CHECKS")
    print("=" * 60)

    # 1. Initial loss
    r1 = check_initial_loss(model, x_train)
    status = "PASS" if r1["passed"] else "FAIL"
    print(f"\n[{status}] Initial loss check")
    print(f"  Initial NLL:  {r1['initial_nll']:.4f}")
    print(f"  Expected NLL: {r1['expected_nll']:.4f}  (standard normal on data)")
    print(f"  Rel. error:   {r1['rel_error']:.4f}")

    # 2. Stable start (trains the model for a few steps)
    r3 = check_stable_start(model, x_train, n_steps=50, lr=1e-3)
    status = "PASS" if r3["passed"] else "FAIL"
    print(f"\n[{status}] Stable start check  (50 steps)")
    print(f"  Loss: {r3['initial_loss']:.4f} -> {r3['final_loss']:.4f}")
    print(f"  NaN/Inf: {r3['has_nan']},  Exploded: {not r3['no_explosion']},  Decreased: {r3['decreased']}")

    # 3. Beats baseline (uses the slightly-trained model from check 2)
    r2 = check_beats_baseline(model, x_train, x_eval)
    status = "PASS" if r2["passed"] else "FAIL"
    print(f"\n[{status}] Beats Gaussian baseline  (after 50 steps)")
    print(f"  Model NLL:    {r2['model_nll']:.4f}")
    print(f"  Baseline NLL: {r2['baseline_nll']:.4f}")
    print(f"  Margin:       {r2['margin']:.4f}")

    print("\n" + "=" * 60)
    all_passed = r1["passed"] and r2["passed"] and r3["passed"]
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print("=" * 60)

    return {
        "initial_loss": r1,
        "beats_baseline": r2,
        "stable_start": r3,
        "all_passed": all_passed,
    }

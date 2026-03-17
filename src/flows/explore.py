"""Data exploration utilities — summary statistics and diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from flows.data import MoonsDataset, MoonsSplits


def summary_stats(ds: MoonsDataset) -> dict:
    """Per-feature and per-class summary statistics."""
    x = ds.x.numpy()
    labels = ds.labels.numpy()
    classes = sorted(np.unique(labels).tolist())

    stats: dict = {
        "name": ds.name,
        "n_samples": len(ds),
        "n_classes": len(classes),
        "class_counts": {int(c): int((labels == c).sum()) for c in classes},
        "global": {
            "mean": x.mean(axis=0).tolist(),
            "std": x.std(axis=0).tolist(),
            "min": x.min(axis=0).tolist(),
            "max": x.max(axis=0).tolist(),
            "median": np.median(x, axis=0).tolist(),
        },
    }
    for c in classes:
        xc = x[labels == c]
        stats[f"class_{int(c)}"] = {
            "mean": xc.mean(axis=0).tolist(),
            "std": xc.std(axis=0).tolist(),
            "min": xc.min(axis=0).tolist(),
            "max": xc.max(axis=0).tolist(),
        }
    return stats


def print_summary(ds: MoonsDataset) -> None:
    """Pretty-print summary statistics for a dataset split."""
    s = summary_stats(ds)
    print(f"=== {s['name']} ({s['n_samples']} samples, {s['n_classes']} classes) ===")
    print(f"  Dataset shape: {ds.x.shape}")
    print(f"  Class counts: {s['class_counts']}")
    g = s["global"]
    print(f"  x1: mean={g['mean'][0]:.3f}, std={g['std'][0]:.3f}, range=[{g['min'][0]:.3f}, {g['max'][0]:.3f}]")
    print(f"  x2: mean={g['mean'][1]:.3f}, std={g['std'][1]:.3f}, range=[{g['min'][1]:.3f}, {g['max'][1]:.3f}]")
    for key in sorted(s):
        if key.startswith("class_") and isinstance(s[key], dict) and "mean" in s[key]:
            c = s[key]
            print(f"  {key}: mean={c['mean']}, std={c['std']}")
    print()


def correlation_matrix(ds: MoonsDataset) -> np.ndarray:
    """Return the 2x2 Pearson correlation matrix."""
    return np.corrcoef(ds.x.numpy(), rowvar=False)


def check_duplicates(ds: MoonsDataset) -> int:
    """Count duplicate rows."""
    x = ds.x.numpy()
    n_unique = len(np.unique(x, axis=0))
    return len(x) - n_unique


def check_nans(ds: MoonsDataset) -> int:
    """Count NaN entries."""
    return int(torch.isnan(ds.x).sum().item())


def split_overlap(a: MoonsDataset, b: MoonsDataset) -> int:
    """Count rows that appear in both datasets (potential leakage)."""
    set_a = set(map(tuple, a.x.numpy().tolist()))
    set_b = set(map(tuple, b.x.numpy().tolist()))
    return len(set_a & set_b)


def ks_test_splits(splits: MoonsSplits) -> dict:
    """2-sample KS test between train and val/test for each feature.

    Returns dict of {comparison: {feature: (statistic, p_value)}}.
    """
    from scipy.stats import ks_2samp

    results = {}
    for other, name in [(splits.val, "train-val"), (splits.test, "train-test")]:
        results[name] = {}
        for i, feat in enumerate(["x1", "x2"]):
            stat, pval = ks_2samp(splits.train.x[:, i].numpy(), other.x[:, i].numpy())
            results[name][feat] = (float(stat), float(pval))
    return results


def print_ks_test(splits: MoonsSplits) -> None:
    """Pretty-print KS test results."""
    results = ks_test_splits(splits)
    print("=== 2-sample KS tests (H0: same distribution) ===")
    for comp, feats in results.items():
        for feat, (stat, pval) in feats.items():
            sig = " ***" if pval < 0.05 else ""
            print(f"  {comp} {feat}: KS={stat:.4f}, p={pval:.4f}{sig}")
    print()


def full_report(splits: MoonsSplits) -> None:
    """Print a comprehensive exploration report across all splits."""
    for ds in [splits.train, splits.val, splits.test]:
        print_summary(ds)
        n_dup = check_duplicates(ds)
        n_nan = check_nans(ds)
        print(f"  Duplicates: {n_dup}, NaNs: {n_nan}")
        corr = correlation_matrix(ds)
        print(f"  Correlation(x1, x2): {corr[0, 1]:.4f}")
        print()

    # KS tests
    print_ks_test(splits)

    # Leakage check
    for a, b, name in [
        (splits.train, splits.val, "train-val"),
        (splits.train, splits.test, "train-test"),
        (splits.val, splits.test, "val-test"),
    ]:
        overlap = split_overlap(a, b)
        print(f"  {name} overlap: {overlap} rows")

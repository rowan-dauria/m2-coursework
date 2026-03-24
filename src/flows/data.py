"""Data loading and management for the moons dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class MoonsDataset:
    """Container for a single split of the moons dataset."""

    x: torch.Tensor  # (N, 2)
    labels: torch.Tensor  # (N,) integer class labels
    name: str

    @classmethod
    def from_csv(cls, path: str | Path, name: str | None = None) -> MoonsDataset:
        path = Path(path)
        raw = np.loadtxt(path, delimiter=",", skiprows=1)
        x = torch.tensor(raw[:, :2], dtype=torch.float32)
        labels = torch.tensor(raw[:, 2], dtype=torch.long)
        return cls(x=x, labels=labels, name=name or path.stem)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __repr__(self) -> str:
        return f"MoonsDataset(name={self.name!r}, n={len(self)})"


@dataclass
class MoonsSplits:
    """All three splits loaded together."""

    train: MoonsDataset
    val: MoonsDataset
    test: MoonsDataset

    @classmethod
    def load(cls, data_dir: str | Path = "data") -> MoonsSplits:
        d = Path(data_dir)
        return cls(
            train=MoonsDataset.from_csv(d / "moons_train.csv", "train"),
            val=MoonsDataset.from_csv(d / "moons_val.csv", "val"),
            test=MoonsDataset.from_csv(d / "moons_test.csv", "test"),
        )

    def all_x(self) -> torch.Tensor:
        return torch.cat([self.train.x, self.val.x, self.test.x])

    def all_labels(self) -> torch.Tensor:
        return torch.cat([self.train.labels, self.val.labels, self.test.labels])

    def normalise(self) -> NormaliseStats:
        """Standardise all splits using training-set statistics.

        Transforms each split in-place so that the training features
        have zero mean and unit variance.  Returns a ``NormaliseStats``
        object that can undo the transformation.
        """
        mean = self.train.x.mean(dim=0)
        std = self.train.x.std(dim=0)

        for ds in (self.train, self.val, self.test):
            ds.x = (ds.x - mean) / std

        return NormaliseStats(mean=mean, std=std)


@dataclass
class NormaliseStats:
    """Stores the mean and std used to standardise the data."""

    mean: torch.Tensor  # (D,)
    std: torch.Tensor  # (D,)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise *x* using the stored statistics."""
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Map standardised *x* back to the original scale."""
        return x * self.std + self.mean

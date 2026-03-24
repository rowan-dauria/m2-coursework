"""Coupling flow model components."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AffineCouplingLayer(nn.Module):
    """Single affine coupling layer for a 2D input h = [h1, h2].

    Forward (base -> data):  h1' = h1,  h2' = h2 * exp(s[h1]) + t[h1]
    Inverse (data -> base):  h1  = h1', h2  = (h2' - t[h1']) * exp(-s[h1'])

    The mask selects which component is fixed (1) vs transformed (0).
    """

    mask: torch.Tensor  # needed to avoid type warning

    def __init__(self, dim, hidden, mask):
        super().__init__()
        # unnecessary for pure CPU work but worth having in case you want to use a GPU
        self.register_buffer("mask", mask)
        # MLP: Linear(D -> H) -> ReLU -> Linear(H -> 2D)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
        )

    def _st(self, h_masked):
        """Run the MLP and return (t, s) with s bounded by tanh."""
        out = self.net(h_masked)
        t, s = out.chunk(2, dim=-1)
        # bound s with tanh for numerical stability
        s = torch.tanh(s)
        return t, s

    def forward(self, h):
        """Forward map f[z] (base -> data).
        Returns (h', log_det_J).
        """
        # zero out features that aren't passed to neural net
        h_masked = h * self.mask
        t, s = self._st(h_masked)

        fixed = h_masked
        # only transform the unmasked components
        transformed = (1 - self.mask) * (h * torch.exp(s) + t)
        # combine the fixed and transformed components
        h_prime = fixed + transformed  # works because fixed/transformed features are mutually exclusive
        # log|det J| = sum of s over transformed (unmasked) dims
        log_det = (s * (1 - self.mask)).sum(dim=-1)
        return h_prime, log_det

    def inverse(self, h_prime):
        """Inverse map f^{-1}[x] (data -> base).
        Returns (h, log_det_J) where log_det_J is for the *inverse* direction.
        """
        h_masked = h_prime * self.mask
        t, s = self._st(h_masked)
        # perform inverse linear transformation to find the latents
        h = h_masked + (1 - self.mask) * ((h_prime - t) * torch.exp(-s))
        # log|det J^{-1}| = -sum of s over transformed dims
        log_det = -(s * (1 - self.mask)).sum(dim=-1)
        return h, log_det


class Flow(nn.Module):
    """Stack of K affine coupling layers with alternating masks.

    Base density: p(z) = N(0, I).
    """

    def __init__(self, dim=2, hidden=128, n_layers=8):
        super().__init__()
        self.dim = dim
        layers = []
        for i in range(n_layers):
            # alternate which component is fixed
            mask = torch.zeros(dim)
            mask[i % 2] = 1.0
            layers.append(AffineCouplingLayer(dim, hidden, mask))
        self.layers = nn.ModuleList(layers)

    def forward(self, z):
        """Forward (sampling) map: z ~ N(0,I) -> x.
        Returns (x, sum of log_det_J).
        """
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in self.layers:
            x, log_det = layer.forward(x)
            log_det_total = log_det_total + log_det
        return x, log_det_total

    def inverse(self, x):
        """Inverse map: x (data) -> z (base).
        Returns (z, sum of log_det_J for the inverse direction).
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def log_prob(self, x):
        """Evaluate log p(x) for a batch of data points."""
        z, log_det_inv = self.inverse(x)
        # log p(z) under standard normal
        log_pz = -0.5 * (self.dim * math.log(2 * math.pi) + (z**2).sum(dim=-1))
        return log_pz + log_det_inv


class SurgeryFlow(Flow):
    """Flow with an appended shear transformation.

    Extends a trained Flow by composing its sampling map f0 with a deterministic
    shear g_alpha, giving a new sampling map f_alpha[z] = g_alpha(f0[z]) and a
    corresponding density p_alpha[x] via the change-of-variables formula.

    The shear map is:
        g_alpha([x1, x2]) = [x1 + alpha * x2, x2]

    whose Jacobian has determinant 1 (log-det = 0) for all alpha.

    Args:
        dim: Dimensionality of the data (must be even).
        hidden: Hidden width of each coupling layer MLP.
        n_layers: Number of coupling layers K (must be even).
        alpha: Shear parameter. alpha=0 recovers the original flow exactly.
    """

    def __init__(self, dim=2, hidden=128, n_layers=8, alpha=0.0):
        # Initialise the parent Flow class with the standard config
        super().__init__(dim=dim, hidden=hidden, n_layers=n_layers)
        self.alpha = alpha

    def forward(self, z):
        # 1. Pass through the trained flow via the parent class forward hook
        x0, log_det_f0 = super().forward(z)

        # 2. Append your deterministic map
        x, log_det_g = self._map(x0, self.alpha)

        # 3. Combine log-determinants
        return x, log_det_f0 + log_det_g

    def inverse(self, x):
        # 1. Pass through the inverse of the deterministic map
        x0, log_det_g_inv = self._inv_map(x, self.alpha)

        # 2. Pass through the trained flow inverse via parent class
        z, log_det_f0_inv = super().inverse(x0)

        # 3. Combine log-determinants
        return z, log_det_g_inv + log_det_f0_inv

    def _map(self, z: torch.Tensor, alpha: float):
        z1, z2 = z.chunk(2, dim=-1)
        # perform shear transformation
        x1 = z1 + alpha * z2
        x2 = z2

        x = torch.cat((x1, x2), dim=-1)
        log_det = torch.zeros(x.shape[0])
        return x, log_det

    def _inv_map(self, x: torch.Tensor, alpha: float):
        x1, x2 = x.chunk(2, dim=1)
        # perform inverse shear transformation
        z1 = x1 - alpha * x2
        z2 = x2

        z = torch.cat((z1, z2), dim=-1)
        log_det = torch.zeros(z.shape[0])
        return z, log_det


def load_surgery_models(
    alpha_values: list[float],
    checkpoint_path: str = "checkpoints/flow_full.pt",
) -> dict[float, SurgeryFlow]:
    """Load a trained checkpoint and build a :class:`SurgeryFlow` for each alpha.

    Returns a dict mapping each alpha value to its eval-mode model.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    models: dict[float, SurgeryFlow] = {}
    for a in alpha_values:
        m = SurgeryFlow(
            dim=config["dim"],
            hidden=config["hidden"],
            n_layers=config["n_layers"],
            alpha=a,
        )
        m.load_state_dict(checkpoint["state_dict"])
        m.eval()
        models[a] = m
    print("Loaded SurgeryFlow for alpha values:", alpha_values)
    return models


def generate_samples(
    models: dict[float, SurgeryFlow],
    n_samples: int = 1000,
    dim: int = 2,
) -> dict[float, torch.Tensor]:
    """Generate samples from each SurgeryFlow model.

    Returns a dict mapping each alpha value to a ``(n_samples, dim)`` tensor.
    """
    samples: dict[float, torch.Tensor] = {}
    with torch.no_grad():
        for a, model in models.items():
            z = torch.randn(n_samples, dim)
            x, _ = model.forward(z)
            samples[a] = x
    return samples

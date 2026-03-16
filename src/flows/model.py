"""Coupling flow model components."""

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

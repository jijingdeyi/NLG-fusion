"""Shared fusion quality scorer Q_θ = (s, σ) (score.md §4.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedScorer(nn.Module):
    """Multi-scale CNN on Concat(IR, VI, F) with score + uncertainty heads.

    Inference metric is only s_F; σ_F is used in preference probability modeling.
    """

    def __init__(
        self,
        in_ch: int = 3,
        channels: tuple[int, ...] = (32, 64, 128, 128),
        sigma_min: float = 0.05,
        sigma_max: float = 2.0,
    ):
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        layers: list[nn.Module] = []
        c_in = in_ch
        for i, c_out in enumerate(channels):
            k = 5 if i == 0 else 3
            p = k // 2
            layers.append(nn.Sequential(
                nn.Conv2d(c_in, c_out, k, stride=2, padding=p),
                nn.ReLU(inplace=True),
            ))
            c_in = c_out
        self.stages = nn.ModuleList(layers)
        feat_dim = sum(2 * c for c in channels)
        self.head_s = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.head_sigma = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hs = []
        z = x
        for stage in self.stages:
            z = stage(z)
            gap = F.adaptive_avg_pool2d(z, 1).flatten(1)
            gmp = F.adaptive_max_pool2d(z, 1).flatten(1)
            hs.append(torch.cat([gap, gmp], dim=1))
        return torch.cat(hs, dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B,3,H,W) → (s, σ) each (B,)."""
        h = self.encode(x)
        s = self.head_s(h).squeeze(-1)
        raw = self.head_sigma(h).squeeze(-1)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw)
        return s, sigma

    def score(self, x: torch.Tensor) -> torch.Tensor:
        s, _ = self.forward(x)
        return s

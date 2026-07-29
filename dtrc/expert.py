"""Per-degradation correction experts operating on C-channel features."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CorrectionExpert(nn.Module):
    """Token-conditioned residual on 16-ch features → 1-ch image residual."""

    def __init__(self, channels: int = 16, token_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.act = nn.GELU()
        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2),
        )
        self.to_res = nn.Conv2d(channels, 1, 1, bias=True)
        nn.init.zeros_(self.to_res.weight)
        nn.init.zeros_(self.to_res.bias)

    def forward(self, features: Tensor, token: Tensor) -> Tensor:
        """
        Args:
            features: [B, C, H, W]
            token: [B, D] or [D]
        Returns:
            residual [B, 1, H, W]
        """
        if token.dim() == 1:
            token = token.unsqueeze(0).expand(features.shape[0], -1)
        z = self.act(self.dw(self.reduce(features)))
        gb = self.token_mlp(token)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + torch.tanh(gamma).view(-1, self.channels, 1, 1)
        beta = 0.1 * torch.tanh(beta).view(-1, self.channels, 1, 1)
        z = gamma * z + beta
        return self.to_res(z)

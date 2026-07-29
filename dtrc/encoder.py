"""Shallow full-resolution encoders for DTRC (C=16, no downsample)."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class DWResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ShallowEncoder(nn.Module):
    """1×H×W → C×H×W."""

    def __init__(self, channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=True),
            nn.GELU(),
            DWResidual(channels),
            DWResidual(channels),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image)

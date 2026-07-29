"""MVP paired degradation synthesis: VI-low, VI-over, IR-blur (single class / sample)."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import Tensor

from dtrc.vocab import IR_MVP_EXPERTS, VI_MVP_EXPERTS


def _gaussian_kernel1d(sigma: float, device, dtype) -> Tensor:
    radius = max(1, int(round(sigma * 3)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coords / max(sigma, 1e-3)) ** 2)
    return kernel / kernel.sum()


def _gaussian_blur(image: Tensor, sigma: float) -> Tensor:
    kh = _gaussian_kernel1d(sigma, image.device, image.dtype).view(1, 1, 1, -1)
    kv = kh.transpose(-1, -2)
    x = image.unsqueeze(0)
    pad = kh.shape[-1] // 2
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    x = F.conv2d(x, kh.expand(1, 1, 1, -1))
    x = F.conv2d(x, kv.expand(1, 1, -1, 1))
    return x.squeeze(0)


def _soft_mask(height: int, width: int) -> Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij',
    )
    cy = random.random() * height
    cx = random.random() * width
    sy = random.uniform(0.15, 0.45) * height
    sx = random.uniform(0.15, 0.45) * width
    dist = ((yy - cy) ** 2) / (sy ** 2 + 1e-6) + ((xx - cx) ** 2) / (sx ** 2 + 1e-6)
    return torch.exp(-0.5 * dist)


def _apply_low(image: Tensor, strength: float) -> Tensor:
    gamma = random.uniform(1.2, 2.2)
    scale = max(0.15, 1.0 - 0.75 * strength)
    return (image.clamp(1e-6, 1.0) ** gamma * scale).clamp(0.0, 1.0)


def _apply_over(image: Tensor, strength: float) -> Tensor:
    return (image * (1.0 + 1.8 * strength) + 0.08 * strength).clamp(0.0, 1.0)


def _apply_blur(image: Tensor, strength: float) -> Tensor:
    return _gaussian_blur(image, sigma=1.0 + 2.5 * strength)


def synthesize_mvp_pair(
    ir: Tensor,
    vi: Tensor,
    clean_prob: float = 0.3,
) -> dict[str, Tensor | str | None]:
    """Return clean/deg pair and GT spatial gates for VI/IR MVP experts.

    Args:
        ir, vi: [1,H,W] in [0,1]
    """
    _, height, width = vi.shape
    vi_gates = torch.zeros(len(VI_MVP_EXPERTS), height, width, dtype=vi.dtype)
    ir_gates = torch.zeros(len(IR_MVP_EXPERTS), height, width, dtype=ir.dtype)
    ir_deg, vi_deg = ir.clone(), vi.clone()
    kind = 'clean'
    modality = 'none'

    if random.random() >= clean_prob:
        # Single degradation, balanced over three experts.
        choice = random.choice(['vi_low', 'vi_over', 'ir_blur'])
        strength = random.uniform(0.45, 0.95)
        mask = _soft_mask(height, width).to(dtype=vi.dtype)
        if choice == 'vi_low':
            kind, modality = 'low', 'vi'
            deg = _apply_low(vi, strength)
            vi_deg = vi * (1.0 - mask) + deg * mask
            vi_gates[VI_MVP_EXPERTS.index('low')] = mask * strength
        elif choice == 'vi_over':
            kind, modality = 'over', 'vi'
            deg = _apply_over(vi, strength)
            vi_deg = vi * (1.0 - mask) + deg * mask
            vi_gates[VI_MVP_EXPERTS.index('over')] = mask * strength
        else:
            kind, modality = 'blur', 'ir'
            deg = _apply_blur(ir, strength)
            ir_deg = ir * (1.0 - mask) + deg * mask
            ir_gates[IR_MVP_EXPERTS.index('blur')] = mask * strength

    return {
        'ir': ir,
        'vi': vi,
        'ir_deg': ir_deg.clamp(0, 1),
        'vi_deg': vi_deg.clamp(0, 1),
        'vi_gates': vi_gates,
        'ir_gates': ir_gates,
        'kind': kind,
        'modality': modality,
    }

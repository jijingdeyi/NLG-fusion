"""Spatially varying synthetic degradations for the VI branch during training.

IR is kept clean (single ``normal`` token); no IR synthesis path.
Each sample uses at most one VI degradation class (``low`` or ``veil``).
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.ldt import VI_DEGRADATION_NAMES

# Regional ops (excluding 'normal').
VI_REGIONAL_DEGRADATIONS = ('low', 'veil')


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


def _random_soft_mask(height: int, width: int) -> Tensor:
    dtype = torch.float32
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=dtype),
        torch.arange(width, dtype=dtype),
        indexing='ij',
    )
    cy = random.random() * height
    cx = random.random() * width
    sy = random.uniform(0.12, 0.42) * height
    sx = random.uniform(0.12, 0.42) * width
    dist = ((yy - cy) ** 2) / (sy ** 2 + 1e-6) + ((xx - cx) ** 2) / (sx ** 2 + 1e-6)
    return torch.exp(-0.5 * dist)


def _apply_low(image: Tensor, strength: float) -> Tensor:
    gamma = random.uniform(1.2, 2.2)
    scale = max(0.15, 1.0 - 0.75 * strength)
    return (image.clamp(1e-6, 1.0) ** gamma * scale).clamp(0.0, 1.0)


def _apply_veil(image: Tensor, strength: float) -> Tensor:
    """Veil token appearance: glare / fog / smoke variants (one class)."""
    mode = random.random()
    if mode < 0.34:
        return (image * (1.0 + 1.8 * strength) + 0.08 * strength).clamp(0.0, 1.0)
    if mode < 0.67:
        t = max(0.35, 1.0 - 0.55 * strength)
        airlight = random.uniform(0.55, 0.95)
        return (image * t + airlight * (1.0 - t)).clamp(0.0, 1.0)
    blurred = _gaussian_blur(image, sigma=1.0 + 2.0 * strength)
    local = F.avg_pool2d(image.unsqueeze(0), kernel_size=7, stride=1, padding=3).squeeze(0)
    contrast = max(0.35, 1.0 - 0.55 * strength)
    lifted = (image - 0.5) * contrast + 0.5 + 0.08 * strength
    return (0.45 * blurred + 0.35 * local + 0.20 * lifted).clamp(0.0, 1.0)


def _apply_vi_degradation(image: Tensor, kind: str, strength: float) -> Tensor:
    ops = {
        'low': _apply_low,
        'veil': _apply_veil,
    }
    if kind not in ops:
        raise KeyError(f'Unknown VI degradation {kind!r}; expected one of {tuple(ops)}')
    return ops[kind](image, strength)


def _clean_pack(vi: Tensor) -> tuple[Tensor, str, Tensor, Tensor]:
    _, height, width = vi.shape
    type_masks = torch.zeros(len(VI_DEGRADATION_NAMES), height, width, dtype=vi.dtype)
    type_masks[0] = 1.0
    union = torch.zeros(1, height, width, dtype=vi.dtype)
    return vi, 'normal', union, type_masks


def synthesize_vi_degradation(
    vi: Tensor,
    apply_prob: float = 1.0,
    min_regions: int = 1,
    max_regions: int = 3,
    global_prob: float = 0.0,
    active_thresh: float = 0.2,
) -> tuple[Tensor, str, Tensor, Tensor]:
    """Returns (degraded, tag, union_mask[1,H,W], type_masks[K,H,W]).

    At most one class ``tag ∈ {normal, low, veil}`` per sample. Regional blobs
    (if any) all use that same class; ``union`` is soft for feathering.
    """
    _, height, width = vi.shape
    name_to_i = {n: i for i, n in enumerate(VI_DEGRADATION_NAMES)}

    if random.random() > apply_prob:
        return _clean_pack(vi)

    kind = random.choice(VI_REGIONAL_DEGRADATIONS)

    if random.random() < global_prob:
        out = _apply_vi_degradation(vi, kind, random.uniform(0.15, 0.55))
        type_masks = torch.zeros(len(VI_DEGRADATION_NAMES), height, width, dtype=vi.dtype)
        type_masks[name_to_i[kind]] = 1.0
        union = torch.ones(1, height, width, dtype=vi.dtype)
        return out.clamp(0.0, 1.0), kind, union, type_masks

    num_regions = random.randint(min_regions, max_regions)
    out = vi.clone()
    claimed = torch.zeros(height, width, dtype=torch.bool)
    union = torch.zeros(1, height, width, dtype=vi.dtype)
    type_masks = torch.zeros(len(VI_DEGRADATION_NAMES), height, width, dtype=vi.dtype)
    type_masks[name_to_i['normal']] = 1.0

    for _ in range(num_regions):
        soft = _random_soft_mask(height, width)
        soft = soft * (~claimed).to(dtype=soft.dtype)
        if float(soft.max()) < active_thresh:
            continue

        strength = random.uniform(0.35, 1.0)
        active = soft > active_thresh
        if not bool(active.any()):
            continue

        soft_s = soft.clamp(0.0, 1.0)
        degraded = _apply_vi_degradation(vi, kind, strength)
        s = soft_s.unsqueeze(0)
        out = torch.where(active.unsqueeze(0), s * degraded + (1.0 - s) * vi, out)

        claimed = claimed | active
        union = torch.maximum(union, s.to(dtype=vi.dtype))
        type_masks[name_to_i['normal']] = type_masks[name_to_i['normal']] * (~active).to(vi.dtype)
        type_masks[name_to_i[kind]] = type_masks[name_to_i[kind]] + active.to(vi.dtype)

    if not bool(claimed.any()):
        return _clean_pack(vi)

    cover = float(union.mean())
    if cover > 0.85 and cover > 1e-6:
        scale = 0.85 / cover
        union = union * scale
        deg_area = claimed.unsqueeze(0)
        out = torch.where(deg_area, scale * out + (1.0 - scale) * vi, out)

    type_masks = type_masks.clamp(min=0.0)
    winner = type_masks.argmax(dim=0)
    type_masks = torch.zeros_like(type_masks)
    type_masks.scatter_(0, winner.unsqueeze(0), 1.0)

    # Tiny blobs: keep spatial mask for the gate, but image-level tag = normal
    # so the global router still sees clean-dominant samples.
    tag = kind if cover >= 0.05 else 'normal'
    return out.clamp(0.0, 1.0), tag, union.to(dtype=vi.dtype), type_masks

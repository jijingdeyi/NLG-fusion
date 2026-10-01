"""Residual-base luminance prior: blend of max and gradient-selected HF."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

PRIOR_BLUR_KS = 5


def _box_blur(image: Tensor, kernel_size: int = PRIOR_BLUR_KS) -> Tensor:
    if kernel_size <= 1:
        return image
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    x = F.pad(image, (pad, pad, pad, pad), mode='reflect')
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1)


def _grad_mag(image: Tensor) -> Tensor:
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx.abs() + dy.abs()


def build_fusion_prior(
    ir: Tensor,
    vi: Tensor,
    hf_alpha: float = 1.0,
    blur_ks: int = PRIOR_BLUR_KS,
) -> Tensor:
    """Bottom map ``m`` for ``u = clip(m + s_r R)``.

    ``hf_alpha=0`` → classical ``max(IR,VI)``;
    ``hf_alpha=1`` → ``LP(max) + grad-selected HF``;
    in between → convex blend.
    """
    alpha = float(max(0.0, min(1.0, hf_alpha)))
    m_int = torch.max(ir, vi)
    if alpha <= 0.0:
        return m_int.clamp(0.0, 1.0)

    lp = _box_blur(m_int, blur_ks)
    hf_ir = ir - _box_blur(ir, blur_ks)
    hf_vi = vi - _box_blur(vi, blur_ks)
    use_vi = (_grad_mag(vi) >= _grad_mag(ir)).to(dtype=ir.dtype)
    hf = use_vi * hf_vi + (1.0 - use_vi) * hf_ir
    m_hf = (lp + hf).clamp(0.0, 1.0)
    if alpha >= 1.0:
        return m_hf
    return ((1.0 - alpha) * m_int + alpha * m_hf).clamp(0.0, 1.0)

"""DTRC losses for stage-2 teacher-forced expert training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_l1(pred: Tensor, target: Tensor, weight: Tensor, outside: float = 0.1) -> Tensor:
    """weight in [0,1]; outside pixels use ``outside`` scale."""
    w = weight.clamp(0.0, 1.0)
    if w.shape[1] != 1 and w.dim() == 4:
        w = w.max(dim=1, keepdim=True).values
    err = (pred - target).abs()
    return (w * err + outside * (1.0 - w) * err).mean()


def rec_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    outside: float = 0.1,
    grad_weight: float = 0.5,
) -> Tensor:
    if mask is None:
        l1 = (pred - target).abs().mean()
    else:
        l1 = masked_l1(pred, target, mask, outside=outside)
    # cheap gradient match
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    g = (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()
    return l1 + grad_weight * g


def fusion_int_tex(
    fused: Tensor,
    ir: Tensor,
    vi: Tensor,
    int_w: float = 0.7,
    tex_w: float = 0.3,
) -> Tensor:
    """Classical fusion terms vs clean sources (not degraded)."""
    int_loss = F.l1_loss(fused, torch.max(ir, vi))
    # max-abs grad proxy
    def _g(x: Tensor) -> Tensor:
        dx = (x[..., :, 1:] - x[..., :, :-1]).abs()
        dy = (x[..., 1:, :] - x[..., :-1, :]).abs()
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return dx + dy

    tex = F.l1_loss(_g(fused), torch.max(_g(ir), _g(vi)))
    return int_w * int_loss + tex_w * tex

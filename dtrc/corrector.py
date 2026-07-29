"""Modality correctors: encode → expert residuals gated by spatial maps."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from dtrc.encoder import ShallowEncoder
from dtrc.expert import CorrectionExpert


class ModalityCorrector(nn.Module):
    """Correct one modality with K degradation experts."""

    def __init__(
        self,
        expert_names: tuple[str, ...],
        channels: int = 16,
        token_dim: int = 64,
    ):
        super().__init__()
        self.expert_names = expert_names
        self.num_experts = len(expert_names)
        self.encoder = ShallowEncoder(channels)
        self.tokens = nn.Parameter(torch.randn(self.num_experts, token_dim) * 0.02)
        self.experts = nn.ModuleList(
            [CorrectionExpert(channels, token_dim) for _ in expert_names]
        )

    def forward(
        self,
        degraded: Tensor,
        spatial_gates: Tensor | None = None,
        enable: Tensor | None = None,
        neutral: bool = False,
    ) -> tuple[Tensor, dict]:
        """
        Args:
            degraded: [B, 1, H, W]
            spatial_gates: [B, K, H, W] in [0,1]; if None and not neutral, identity gates=0
            enable: [B, K] or [K] multipliers (manual one-hot / auto ones); default all-1
            neutral: force zero correction
        Returns:
            corrected image, aux dict
        """
        batch = degraded.shape[0]
        device, dtype = degraded.device, degraded.dtype
        feats = self.encoder(degraded)

        if neutral or self.num_experts == 0:
            return degraded, {
                'residual': degraded.new_zeros(batch, 1, *degraded.shape[-2:]),
                'gates': degraded.new_zeros(batch, max(self.num_experts, 1), *degraded.shape[-2:]),
            }

        if spatial_gates is None:
            spatial_gates = degraded.new_zeros(batch, self.num_experts, *degraded.shape[-2:])
        if enable is None:
            enable = degraded.new_ones(batch, self.num_experts)
        elif enable.dim() == 1:
            enable = enable.view(1, -1).expand(batch, -1)

        total = degraded.new_zeros(batch, 1, *degraded.shape[-2:])
        per_res = []
        for k, expert in enumerate(self.experts):
            tok = self.tokens[k:k + 1].expand(batch, -1)
            r_k = expert(feats, tok)
            g = enable[:, k].view(batch, 1, 1, 1) * spatial_gates[:, k:k + 1]
            total = total + g * r_k
            per_res.append(r_k)

        corrected = (degraded + total).clamp(0.0, 1.0)
        return corrected, {
            'residual': total,
            'gates': spatial_gates,
            'per_residual': torch.stack(per_res, dim=1) if per_res else None,
            'enable': enable,
        }


class DualModalityCorrector(nn.Module):
    """VI + IR correctors for the MVP vocabulary."""

    def __init__(
        self,
        vi_experts: tuple[str, ...] = ('low', 'over'),
        ir_experts: tuple[str, ...] = ('blur',),
        channels: int = 16,
        token_dim: int = 64,
    ):
        super().__init__()
        self.vi = ModalityCorrector(vi_experts, channels, token_dim)
        self.ir = ModalityCorrector(ir_experts, channels, token_dim)

    def forward(
        self,
        ir_deg: Tensor,
        vi_deg: Tensor,
        ir_gates: Tensor | None = None,
        vi_gates: Tensor | None = None,
        ir_enable: Tensor | None = None,
        vi_enable: Tensor | None = None,
        neutral: bool = False,
    ) -> tuple[Tensor, Tensor, dict]:
        ir_hat, ir_aux = self.ir(ir_deg, ir_gates, ir_enable, neutral=neutral)
        vi_hat, vi_aux = self.vi(vi_deg, vi_gates, vi_enable, neutral=neutral)
        return ir_hat, vi_hat, {'ir': ir_aux, 'vi': vi_aux}

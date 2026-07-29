"""Legacy LDT token bank / router / SRC (no longer wired into ``modules.fuser.Fuser``).

Kept for reference and historical scripts. New control path: ``dtrc/``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

VI_DEGRADATION_PROMPTS = (
    'a normal visible image',
    'a visible image with low light degradation',
    'a visible image with veiling degradation from overexposure glare or haze smoke',
)

IR_DEGRADATION_PROMPTS = (
    'a normal infrared image',
)

# Visible: low light vs veiling (overexposure / haze / smoke share one fusion token).
# Infrared: kept clean in training (single normal token).
VI_DEGRADATION_NAMES = ('normal', 'low', 'veil')
IR_DEGRADATION_NAMES = ('normal',)

SRC_MODES = ('src', 'global_film', 'residual_film')
# src gate: typed K-way softmax M_k vs binary degradation union S (where only).
GATE_MODES = ('typed', 'union')


def image_statistics(image: Tensor) -> Tensor:
    """Per-image stats [B, 6]: mean, std, q10, q90, mean|grad|, overexposure ratio."""
    flat = image.flatten(1)
    mu = flat.mean(dim=1, keepdim=True)
    sigma = flat.std(dim=1, keepdim=True)
    q10 = torch.quantile(flat, 0.1, dim=1, keepdim=True)
    q90 = torch.quantile(flat, 0.9, dim=1, keepdim=True)

    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    grad_mag = (dx.abs().mean(dim=(2, 3)) + dy.abs().mean(dim=(2, 3))) * 0.5
    grad_mag = grad_mag.reshape(image.shape[0], 1)

    r_over = (image > 0.95).float().flatten(1).mean(dim=1, keepdim=True)
    return torch.cat([mu, sigma, q10, q90, grad_mag, r_over], dim=1)


class DegradationTokenBank(nn.Module):
    """CLIP-initialized, task-adapted degradation tokens (VI + IR prompt banks)."""

    def __init__(
        self,
        token_dim: int = 64,
        clip_dim: int = 512,
        num_vi: int | None = None,
        num_ir: int | None = None,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_vi = num_vi or len(VI_DEGRADATION_PROMPTS)
        self.num_ir = num_ir or len(IR_DEGRADATION_PROMPTS)
        self.proj = nn.Linear(clip_dim, token_dim, bias=False)
        self.tokens_vi = nn.Parameter(torch.zeros(self.num_vi, token_dim))
        self.tokens_ir = nn.Parameter(torch.zeros(self.num_ir, token_dim))
        self.register_buffer('tokens_vi_init', torch.zeros(self.num_vi, token_dim))
        self.register_buffer('tokens_ir_init', torch.zeros(self.num_ir, token_dim))

    def initialize_from_clip(self, clip_vi: Tensor, clip_ir: Tensor) -> None:
        """clip_*: [K, clip_dim] L2-normalized CLIP text features.

        ``proj`` is used only here (offline), then frozen so it is not a
        dangling trainable parameter during fusion training.
        """
        with torch.no_grad():
            vi = self.proj(clip_vi)
            ir = self.proj(clip_ir)
            self.tokens_vi.copy_(vi)
            self.tokens_ir.copy_(ir)
            self.tokens_vi_init.copy_(vi)
            self.tokens_ir_init.copy_(ir)
        for param in self.proj.parameters():
            param.requires_grad = False

    def mix_vi(self, alpha: Tensor) -> Tensor:
        return alpha @ self.tokens_vi

    def mix_ir(self, alpha: Tensor) -> Tensor:
        return alpha @ self.tokens_ir

    def token_regularization_loss(self) -> Tensor:
        return (
            (self.tokens_vi - self.tokens_vi_init).pow(2).sum()
            + (self.tokens_ir - self.tokens_ir_init).pow(2).sum()
        )

    def token_separation_loss(self) -> Tensor:
        """Penalize high cosine similarity between distinct degradation tokens."""
        # eps avoids NaN grads from normalize when a token collapses near zero.
        vi = F.normalize(self.tokens_vi, dim=1, eps=1e-6)
        ir = F.normalize(self.tokens_ir, dim=1, eps=1e-6)
        eye_vi = torch.eye(self.num_vi, device=vi.device, dtype=vi.dtype)
        eye_ir = torch.eye(self.num_ir, device=ir.device, dtype=ir.dtype)
        off_vi = (vi @ vi.T - eye_vi).pow(2).mean()
        off_ir = (ir @ ir.T - eye_ir).pow(2).mean()
        return off_vi + off_ir


class TokenRouter(nn.Module):
    """Lightweight CNN + image statistics -> softmax token weights."""

    def __init__(self, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(16 + 6, 32),
            nn.GELU(),
            nn.Linear(32, num_tokens),
        )

    def forward_logits(self, image: Tensor) -> Tensor:
        feat = self.conv(image)
        feat = feat.mean(dim=(2, 3))
        stats = image_statistics(image)
        return self.mlp(torch.cat([feat, stats], dim=1))

    def forward(self, image: Tensor) -> Tensor:
        return F.softmax(self.forward_logits(image), dim=-1)


class SpatialResidualCalibration(nn.Module):
    """SRC: FiLM deltas with spatial gates.

    ``gate_mode='typed'`` (legacy):
        M̂ = softmax(Conv(lowpass(B)))_k
        B' = B + η · Σ_k M_k ⊙ Δ(t_k)

    ``gate_mode='union'`` (default): binary where + α for what
        Ŝ = σ(Conv(lowpass(B)))
        Δ_α = Σ_k α_k Δ(t_k)
        B' = B + η · S ⊙ Δ_α
    S may be GT union ``1-M_normal`` or Ŝ (``pred_mask_prob``). M/S detached
    from L_fus by default; FiLM Δ stays live.
    """

    def __init__(
        self,
        in_channels: int,
        token_dim: int,
        num_tokens: int,
        eta: float = 0.5,
        modulation_scale: float = 1.0,
        src_mode: str = 'src',
        detach_from_fusion: bool = True,
        pred_mask_prob: float = 0.0,
        gate_pool: int = 8,
        gate_mode: str = 'union',
    ):
        super().__init__()
        if src_mode not in SRC_MODES:
            raise ValueError(f'src_mode must be one of {SRC_MODES}, got {src_mode!r}')
        if gate_mode not in GATE_MODES:
            raise ValueError(f'gate_mode must be one of {GATE_MODES}, got {gate_mode!r}')
        self.src_mode = src_mode
        self.gate_mode = gate_mode
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.detach_from_fusion = detach_from_fusion
        # Back-compat alias used by Fuser.detach_src_gate setter.
        self.detach_gate_from_fusion = detach_from_fusion
        self.pred_mask_prob = float(pred_mask_prob)
        self.gate_pool = max(1, int(gate_pool))
        self.channel_mlp = nn.Sequential(
            nn.Linear(token_dim, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, in_channels * 2),
        )
        gate_out = 1 if gate_mode == 'union' else num_tokens
        # Gate from low-pass features only (avoids raw texture shortcuts).
        self.gate_stem = (
            nn.Sequential(
                nn.Conv2d(in_channels + 2, 32, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, gate_out, kernel_size=1),
            )
            if src_mode == 'src'
            else None
        )
        # Typed-only token bias; unused in union mode.
        self.token_gate_bias = (
            nn.Linear(token_dim, 1) if src_mode == 'src' and gate_mode == 'typed' else None
        )
        # Kept for old-checkpoint compatibility; no longer added into gate logits.
        self.delta_to_gate = (
            nn.Conv2d(in_channels, 1, kernel_size=1) if src_mode == 'src' else None
        )
        # Legacy name kept for older references / viz checks.
        self.spatial_gate = self.gate_stem
        self.eta = nn.Parameter(torch.tensor(float(eta)))
        self.modulation_scale = nn.Parameter(torch.tensor(float(modulation_scale)))
        self._init_channel_mlp()
        if self.token_gate_bias is not None:
            nn.init.zeros_(self.token_gate_bias.bias)
            nn.init.normal_(self.token_gate_bias.weight, std=0.05)
        if self.delta_to_gate is not None:
            nn.init.normal_(self.delta_to_gate.weight, std=0.02)
            if self.delta_to_gate.bias is not None:
                nn.init.zeros_(self.delta_to_gate.bias)

    def _init_channel_mlp(self) -> None:
        for layer in self.channel_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def film_coefficients(self, token: Tensor) -> tuple[Tensor, Tensor]:
        """Return (gamma, beta) as [B, C, 1, 1] for diagnostics / diversity loss."""
        batch = token.shape[0]
        modulation = self.channel_mlp(token)
        gamma_raw, beta_raw = modulation.view(batch, -1, 1, 1).chunk(2, dim=1)
        scale = self.modulation_scale
        gamma = 1.0 + scale * torch.tanh(gamma_raw)
        beta = scale * torch.tanh(beta_raw)
        return gamma, beta

    def film_diversity_loss(self, tokens: Tensor, margin: float = 0.1) -> Tensor:
        """Hinge on pairwise ||(γ,β)_i - (γ,β)_j|| for a bank of tokens [K, D]."""
        gamma, beta = self.film_coefficients(tokens)
        params = torch.cat([gamma.flatten(1), beta.flatten(1)], dim=1)
        k = params.shape[0]
        if k < 2:
            return params.new_zeros(())
        diff = params.unsqueeze(0) - params.unsqueeze(1)
        dist_sq = diff.pow(2).sum(dim=-1)
        ii, jj = torch.triu_indices(k, k, offset=1, device=params.device)
        dist = (dist_sq[ii, jj] + 1e-8).sqrt()
        return F.relu(margin - dist).mean()

    def _unit_gate(self, features: Tensor, num_tokens: int | None = None) -> Tensor:
        k = num_tokens or self.num_tokens
        return torch.ones(
            features.shape[0], k, features.shape[2], features.shape[3],
            device=features.device, dtype=features.dtype,
        )

    def _lowpass_for_gate(self, features: Tensor) -> Tensor:
        """Avg-pool then upsample so gate logits prefer smooth spatial structure."""
        if self.gate_pool <= 1:
            return features
        height, width = features.shape[2], features.shape[3]
        pooled = F.avg_pool2d(features, kernel_size=self.gate_pool, stride=self.gate_pool)
        return F.interpolate(pooled, size=(height, width), mode='bilinear', align_corners=False)

    def forward(
        self,
        features: Tensor,
        token: Tensor,
        eta_scale: float = 1.0,
        tokens: Tensor | None = None,
        type_masks: Tensor | None = None,
        alpha: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Calibrate features.

        Args:
            features: [B, C, H, W]
            token: mixed token [B, D] (used only for global_film / residual_film)
            tokens: full bank [K, D] for typed SRC (required when src_mode='src')
            type_masks: optional GT [B, K, H, W] applied in the residual (train)
            alpha: optional [B, K] broadcast as spatial-constant gates (manual)
        Returns:
            calibrated features, predicted gate:
            - union: Ŝ [B, 1, H, W] degradation mass
            - typed: M̂ [B, K, H, W]
            (or unit gates for non-src modes)
        """
        if self.src_mode == 'global_film':
            gamma, beta = self.film_coefficients(token)
            return gamma * features + beta, self._unit_gate(features, 1)

        if self.src_mode == 'residual_film':
            gamma, beta = self.film_coefficients(token)
            delta = gamma * features + beta - features
            calibrated = features + (self.eta * eta_scale) * delta
            return calibrated, self._unit_gate(features, 1)

        if tokens is None:
            raise ValueError("typed SRC requires 'tokens' [K, D] bank")
        if self.gate_mode == 'union':
            return self._forward_union(
                features, tokens, eta_scale=eta_scale,
                type_masks=type_masks, alpha=alpha,
            )
        return self._forward_typed(
            features, tokens, eta_scale=eta_scale,
            type_masks=type_masks, alpha=alpha,
        )

    def _gate_logits(self, features: Tensor) -> Tensor:
        feat_lp = self._lowpass_for_gate(features)
        avg_c = feat_lp.mean(dim=1, keepdim=True)
        max_c, _ = feat_lp.max(dim=1, keepdim=True)
        return self.gate_stem(torch.cat([feat_lp, avg_c, max_c], dim=1))

    def _token_deltas(self, features: Tensor, tokens: Tensor) -> list[Tensor]:
        batch = features.shape[0]
        deltas: list[Tensor] = []
        for k in range(tokens.shape[0]):
            tok = tokens[k:k + 1].expand(batch, -1)
            gamma, beta = self.film_coefficients(tok)
            deltas.append(gamma * features + beta - features)
        return deltas

    def _maybe_pred_spatial(self, pred: Tensor, gt: Tensor) -> Tensor:
        use_pred = (
            self.training
            and self.pred_mask_prob > 0.0
            and float(torch.rand((), device=pred.device)) < self.pred_mask_prob
        )
        return pred if use_pred else gt

    def _forward_union(
        self,
        features: Tensor,
        tokens: Tensor,
        eta_scale: float = 1.0,
        type_masks: Tensor | None = None,
        alpha: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Binary where-gate S; what from α-mixed FiLM deltas."""
        batch = features.shape[0]
        num_tokens = tokens.shape[0]
        shat = torch.sigmoid(self._gate_logits(features))

        if alpha is None:
            alpha = features.new_full((batch, num_tokens), 1.0 / num_tokens)
        alpha = alpha.to(device=features.device, dtype=features.dtype)
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)
        alpha_b = alpha.view(batch, num_tokens, 1, 1)

        deltas = self._token_deltas(features, tokens)
        delta = torch.zeros_like(features)
        for k in range(num_tokens):
            delta = delta + alpha_b[:, k:k + 1] * deltas[k]

        if type_masks is not None:
            # Soft feathered where from union channel if present as soft; else hard.
            # Teacher uses hard occupancy 1-M_normal (exclusive classes).
            s_gt = (1.0 - type_masks[:, 0:1]).clamp(0.0, 1.0)
            s_apply = self._maybe_pred_spatial(shat, s_gt)
        else:
            s_apply = shat

        detach = self.detach_from_fusion or self.detach_gate_from_fusion
        s_use = s_apply.detach() if detach else s_apply
        calibrated = features + (self.eta * eta_scale) * (s_use * delta)
        return calibrated, shat

    def _forward_typed(
        self,
        features: Tensor,
        tokens: Tensor,
        eta_scale: float = 1.0,
        type_masks: Tensor | None = None,
        alpha: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch = features.shape[0]
        num_tokens = tokens.shape[0]
        height, width = features.shape[2], features.shape[3]

        logits = self._gate_logits(features)
        if self.token_gate_bias is not None:
            logits = logits + self.token_gate_bias(tokens).view(1, num_tokens, 1, 1)

        deltas = self._token_deltas(features, tokens)
        mhat = torch.softmax(logits, dim=1)

        if type_masks is not None:
            m_apply = self._maybe_pred_spatial(mhat, type_masks)
        elif alpha is not None:
            m_apply = alpha[:, :, None, None].expand(-1, -1, height, width)
        else:
            m_apply = mhat

        residual = torch.zeros_like(features)
        detach = self.detach_from_fusion or self.detach_gate_from_fusion
        m_use = m_apply.detach() if detach else m_apply
        for k in range(num_tokens):
            residual = residual + m_use[:, k:k + 1] * deltas[k]
        calibrated = features + (self.eta * eta_scale) * residual
        return calibrated, mhat

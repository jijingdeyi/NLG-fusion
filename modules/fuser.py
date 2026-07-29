"""Lightweight IR–VI fusion backbone (tri-phase attention + dilated decoder).

Decoder input: ``[B_ir; B_vi]`` only (no ``m``).
Composition: ``u = clip(m + s_r R)``; ``m`` blends max and max_hf via ``hf_alpha``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from ldt.defaults import DEFAULT_PRIOR_HF_ALPHA
from ldt.fusion_prior import build_fusion_prior
from modules.layers.conv_group import ConvGroup
from modules.layers.d_group import DGroup
from modules.layers.transformer import TransformerBlock

_BACKBONE_KEY_PREFIXES = (
    'att_a_conv.',
    'att_b_conv.',
    'trans_base.',
    'trans_a.',
    'trans_b.',
    'decoder.',
    'residual_scale',
)


class Fuser(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        use_bn: bool = False,
        residual_scale: float = 0.25,
        hf_alpha: float = DEFAULT_PRIOR_HF_ALPHA,
        **_ignored,
    ):
        super().__init__()
        self.hf_alpha = float(hf_alpha)
        self.att_a_conv = ConvGroup(nn.Conv2d(4, 1, kernel_size=3, padding='same', bias=False), use_bn=True)
        self.att_b_conv = ConvGroup(nn.Conv2d(4, 1, kernel_size=3, padding='same', bias=False), use_bn=True)
        self.trans_base = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.trans_a = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.trans_b = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.decoder = DGroup(in_c=6, out_c=1, dim=dim, k_size=3, use_bn=use_bn)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def build_prior(self, ir: Tensor, vi: Tensor) -> Tensor:
        return build_fusion_prior(ir, vi, hf_alpha=self.hf_alpha)

    def forward(self, i_in: Tensor, return_aux: bool = False, **_ignored):
        i_ir, i_vi = torch.chunk(i_in, 2, dim=1)
        m = self.build_prior(i_ir, i_vi)
        res_ir, attn_ir = self._attention(self.att_a_conv, self.trans_a, i_ir, m)
        res_vi, attn_vi = self._attention(self.att_b_conv, self.trans_b, i_vi, m)
        b_ir = torch.cat([i_ir * attn_ir, res_ir], dim=1)
        b_vi = torch.cat([i_vi * attn_vi, res_vi], dim=1)
        residual = self.decoder(torch.cat([b_ir, b_vi], dim=1))
        fused = (m + self.residual_scale * residual).clamp(0.0, 1.0)
        if return_aux:
            return {
                'fused': fused,
                'residual': residual,
                'prior': m,
                'b_ir': b_ir,
                'b_vi': b_vi,
            }
        return fused

    def _attention(self, att_conv, trans_detail, i_a, m):
        z = torch.cat([i_a, m], dim=1)
        z_max, _ = torch.max(z, dim=1, keepdim=True)
        z_avg = torch.mean(z, dim=1, keepdim=True)
        z_detail = trans_detail(z)
        z_base = self.trans_base(z)
        attn = torch.sigmoid(att_conv(torch.cat([z_max, z_avg, z_detail], dim=1)))
        return z_base, attn


def extract_backbone_state_dict(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        name = key[len('fuser.'):] if key.startswith('fuser.') else key
        if any(name == p or name.startswith(p) for p in _BACKBONE_KEY_PREFIXES):
            out[name] = value
    return out


def load_backbone_from_checkpoint(
    ckpt_path: str | Path,
    *,
    dim: int | None = None,
    freeze: bool = False,
    map_location='cpu',
) -> Fuser:
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    state = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw
    hparams = raw.get('hyper_parameters') or raw.get('hparams') or {}
    args = hparams.get('args') if isinstance(hparams, dict) else None
    if dim is None:
        dim = int(getattr(args, 'dim', 64) if args is not None else 64)
    hf_alpha = float(
        getattr(args, 'prior_hf_alpha', DEFAULT_PRIOR_HF_ALPHA) if args is not None
        else DEFAULT_PRIOR_HF_ALPHA
    )

    fuser = Fuser(dim=dim, use_bn=False, hf_alpha=hf_alpha)
    bb = extract_backbone_state_dict(state)
    model_sd = fuser.state_dict()
    filtered = {
        k: v for k, v in bb.items()
        if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)
    }
    missing, _ = fuser.load_state_dict(filtered, strict=False)
    print(f'[backbone] loaded {len(filtered)}/{len(bb)} tensors from {ckpt_path}')
    if missing:
        print(f'[backbone] missing (re-init): {len(missing)} keys')
    if freeze:
        for p in fuser.parameters():
            p.requires_grad = False
        fuser.eval()
    return fuser

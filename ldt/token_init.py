"""Offline CLIP encoding for degradation token initialization (run once before training)."""

from __future__ import annotations

from pathlib import Path

import clip
import torch

from modules.ldt import IR_DEGRADATION_PROMPTS, VI_DEGRADATION_PROMPTS


def encode_prompts(model_clip, prompts: tuple[str, ...], device: str = 'cuda') -> torch.Tensor:
    tokens = clip.tokenize(list(prompts)).to(device)
    with torch.no_grad():
        feats = model_clip.encode_text(tokens).float()
    return feats / feats.norm(dim=-1, keepdim=True)


def build_degradation_token_init(device: str = 'cuda') -> dict[str, torch.Tensor]:
    model_clip, _ = clip.load('ViT-B/32', device=device)
    model_clip.eval()
    clip_vi = encode_prompts(model_clip, VI_DEGRADATION_PROMPTS, device)
    clip_ir = encode_prompts(model_clip, IR_DEGRADATION_PROMPTS, device)
    return {'clip_vi': clip_vi.cpu(), 'clip_ir': clip_ir.cpu()}


def save_degradation_token_init(path: str | Path, device: str = 'cuda') -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_degradation_token_init(device=device)
    torch.save(payload, path)
    return path


def load_degradation_token_init(path: str | Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')

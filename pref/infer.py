"""Single-candidate scoring and S_val checkpoint selection (score.md §4.5)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch

from pref.dataset import load_triple
from pref.model import SharedScorer


def load_scorer(ckpt_path: Path | str, device: str = 'cpu') -> tuple[SharedScorer, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SharedScorer().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


@torch.no_grad()
def score_fusion(
    model: SharedScorer,
    ir: str,
    vi: str,
    fusion: str,
    size: int = 128,
    device: str = 'cpu',
) -> float:
    x = load_triple(ir, vi, fusion, size=size).unsqueeze(0).to(device)
    return float(model.score(x).item())


@torch.no_grad()
def score_scene_candidates(
    model: SharedScorer,
    ir: str,
    vi: str,
    candidates: dict[str, str],
    size: int = 128,
    device: str = 'cpu',
) -> dict[str, float]:
    """O(K) scoring for all candidates under one source pair."""
    out = {}
    for name, path in candidates.items():
        out[name] = score_fusion(model, ir, vi, path, size=size, device=device)
    return out


@torch.no_grad()
def S_val(
    model: SharedScorer,
    scenes: Sequence[dict],
    size: int = 128,
    device: str = 'cpu',
) -> float:
    """Mean score of fused results on a fixed validation set.

    Each scene dict needs: ir, vi, fusion (path of G(ir,vi)).
    """
    if not scenes:
        return float('nan')
    scores = [
        score_fusion(model, s['ir'], s['vi'], s['fusion'], size=size, device=device)
        for s in scenes
    ]
    return float(sum(scores) / len(scores))


def select_checkpoint(
    model: SharedScorer,
    candidates: Iterable[tuple[str, Sequence[dict]]],
    size: int = 128,
    device: str = 'cpu',
) -> tuple[str, float, dict[str, float]]:
    """G* = argmax_t S_val(G_t).

    `candidates`: iterable of (name, scene_list).
    """
    scores = {}
    for name, scenes in candidates:
        scores[name] = S_val(model, scenes, size=size, device=device)
    best = max(scores, key=scores.get)
    return best, scores[best], scores

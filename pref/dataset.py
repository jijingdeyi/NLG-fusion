"""Pair dataset that yields (x_A, x_B, y, R) (score.md §4.1 / §4.4).

Training samples are pair-level only: Annotations are never expanded into
duplicate training rows — they only determine (y, R) upstream.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def load_gray(path: str | Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_triple(
    ir: str,
    vi: str,
    fusion: str,
    size: int = 224,
    input_mode: str = 'ir_vis_f',
) -> torch.Tensor:
    """Build a 3-channel tensor for ResNet-18.

    Modes (all return C=3 for a shared stem):
      ir_vis_f: Concat(IR, VIS, F)          — full source-conditioned
      f_only:   Concat(F, F, F)
      ir_f:     Concat(IR, F, F)
      vis_f:    Concat(VIS, F, F)
    """
    f = load_gray(fusion, size).astype(np.float32) / 255.0
    if input_mode == 'f_only':
        x = np.stack([f, f, f], axis=0)
    elif input_mode == 'ir_f':
        ir_img = load_gray(ir, size).astype(np.float32) / 255.0
        x = np.stack([ir_img, f, f], axis=0)
    elif input_mode == 'vis_f':
        vi_img = load_gray(vi, size).astype(np.float32) / 255.0
        x = np.stack([vi_img, f, f], axis=0)
    elif input_mode == 'ir_vis_f':
        ir_img = load_gray(ir, size).astype(np.float32) / 255.0
        vi_img = load_gray(vi, size).astype(np.float32) / 255.0
        x = np.stack([ir_img, vi_img, f], axis=0)
    else:
        raise ValueError(f'unknown input_mode {input_mode}')
    return torch.from_numpy(x)


def geometric_augment(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    *,
    crop_scale: tuple[float, float] = (0.8, 1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identical crop + horizontal flip on both triples (IR/VI/F stay aligned).

    No photometric / brightness-contrast jitter — those would break fusion cues.
    """
    _, h, w = x_a.shape
    # synced random crop → resize back
    scale = float(torch.empty(()).uniform_(crop_scale[0], crop_scale[1]).item())
    ch = max(8, int(round(h * scale)))
    cw = max(8, int(round(w * scale)))
    top = int(torch.randint(0, h - ch + 1, (1,)).item()) if ch < h else 0
    left = int(torch.randint(0, w - cw + 1, (1,)).item()) if cw < w else 0
    x_a = x_a[:, top:top + ch, left:left + cw]
    x_b = x_b[:, top:top + ch, left:left + cw]
    x_a = torch.nn.functional.interpolate(
        x_a.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False,
    ).squeeze(0)
    x_b = torch.nn.functional.interpolate(
        x_b.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False,
    ).squeeze(0)
    if torch.rand(()) < 0.5:
        x_a = torch.flip(x_a, dims=[-1])
        x_b = torch.flip(x_b, dims=[-1])
    return x_a, x_b


def is_control_pair(row: dict) -> bool:
    """Obvious degradation / gold controls — useful but must not dominate batches."""
    if row.get('is_gold'):
        return True
    tags = f"{row.get('candidate_a','')}|{row.get('candidate_b','')}|{row.get('left','')}|{row.get('right','')}"
    return any(t in tags for t in ('deg_', 'gold_'))


class PreferencePairDataset(Dataset):
    """Each item: supervision row with path_a / path_b / y / R / ir / vi."""

    def __init__(
        self,
        rows: Sequence[dict],
        size: int = 224,
        augment: bool = False,
        exclude_gold: bool = True,
        exclude_control: bool = False,
        input_mode: str = 'ir_vis_f',
    ):
        self.size = size
        self.augment = augment
        self.input_mode = input_mode
        self.rows = []
        for r in rows:
            if exclude_gold and r.get('is_gold'):
                continue
            if exclude_control and is_control_pair(r):
                continue
            if r.get('path_a') is None or r.get('path_b') is None:
                continue
            if r.get('ir') is None or r.get('vi') is None:
                continue
            self.rows.append(r)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        x_a = load_triple(r['ir'], r['vi'], r['path_a'], self.size, self.input_mode)
        x_b = load_triple(r['ir'], r['vi'], r['path_b'], self.size, self.input_mode)
        if self.augment:
            x_a, x_b = geometric_augment(x_a, x_b)
        y = torch.tensor([r['y_A'], r['y_T'], r['y_B']], dtype=torch.float32)
        R = torch.tensor(float(r['R']), dtype=torch.float32)
        return x_a, x_b, y, R

    def scene_sample_weights(self, control_downweight: float = 0.35) -> list[float]:
        """P(pair) ∝ 1/n_pairs(scene), with optional downweight on control pairs.

        Equivalent to: pick a scene uniformly, then a pair inside it uniformly
        (before control reweighting). Prevents pair-heavy scenes from dominating.
        """
        counts = Counter(str(r.get('scene_id') or '') for r in self.rows)
        weights = []
        for r in self.rows:
            sid = str(r.get('scene_id') or '')
            w = 1.0 / max(counts[sid], 1)
            if is_control_pair(r):
                w *= float(control_downweight)
            weights.append(w)
        return weights


def make_scene_loader(
    ds: PreferencePairDataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 2,
    control_downweight: float = 0.35,
    seed: int = 443,
) -> torch.utils.data.DataLoader:
    from torch.utils.data import DataLoader

    if not shuffle or len(ds) == 0:
        return DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    weights = ds.scene_sample_weights(control_downweight=control_downweight)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(ds),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )

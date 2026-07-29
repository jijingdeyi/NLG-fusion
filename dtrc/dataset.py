"""Dataset wrapper: clean LLVIP pairs + MVP DTRC synthesis."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from dtrc.synth import synthesize_mvp_pair
from ldt.dataset import IVIFDataset


class DTRCPairDataset(Dataset):
    """Yields clean + degraded tensors and GT gates for stage-2 training."""

    def __init__(
        self,
        root: str | Path,
        mode: str = 'train',
        crop_size: int = 320,
        val_samples: int = 30,
        clean_prob: float = 0.3,
    ):
        self.base = IVIFDataset(
            root=root, mode=mode, crop_size=crop_size, val_samples=val_samples,
            vi_synth=False, vi_synth_prob=0.0, synth_global_prob=0.0,
        )
        self.clean_prob = clean_prob
        self.mode = mode

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index]
        ir = sample['ir']  # [1,H,W]
        vi = sample['vi']
        # Prefer clean sources if dataset already attached *_clean.
        ir_c = sample.get('ir_clean', ir)
        vi_c = sample.get('vi_clean', vi)

        if self.mode == 'val':
            # Deterministic-ish: still random but always apply one deg for gain metrics
            # when not clean — use fixed clean_prob 0.2
            pack = synthesize_mvp_pair(ir_c, vi_c, clean_prob=0.2)
        else:
            pack = synthesize_mvp_pair(ir_c, vi_c, clean_prob=self.clean_prob)

        return {
            'ir': pack['ir'],
            'vi': pack['vi'],
            'ir_deg': pack['ir_deg'],
            'vi_deg': pack['vi_deg'],
            'ir_gates': pack['ir_gates'],
            'vi_gates': pack['vi_gates'],
            'kind': pack['kind'],
            'modality': pack['modality'],
            'name': sample.get('name', str(index)),
        }

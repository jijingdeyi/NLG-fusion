import random
from pathlib import Path
from typing import List

import cv2
import torch
import torch.nn.functional as F
from kornia import image_to_tensor
from torch import Tensor
from torch.utils.data import Dataset

from ldt.degradation_synthesis import synthesize_vi_degradation
from ldt.defaults import DEFAULT_VAL_SAMPLES
from modules.ldt import VI_DEGRADATION_NAMES

# Legacy VI synth (optional; backbone train disables it). Kept for DTRC / experiments.
DEFAULT_SYNTH_GLOBAL_PROB = 0.0

_IMAGE_SUFFIXES = {'.jpg', '.bmp', '.png'}
_SPLIT_SEED = 443
_TRAIN_RATIO = 0.97


def list_paired_samples(root: Path) -> List[str]:
    """Return sorted filenames present in both ir/ and vi/."""
    ir_dir, vi_dir = root / 'ir', root / 'vi'
    if not ir_dir.is_dir() or not vi_dir.is_dir():
        raise FileNotFoundError(f'expected directories: {ir_dir} and {vi_dir}')

    ir_names = {p.name for p in ir_dir.glob('*') if p.suffix in _IMAGE_SUFFIXES}
    vi_names = {p.name for p in vi_dir.glob('*') if p.suffix in _IMAGE_SUFFIXES}
    pairs = sorted(ir_names & vi_names)
    if not pairs:
        raise FileNotFoundError(f'no paired images under {ir_dir} and {vi_dir}')
    return pairs


def resolve_samples(root: Path, mode: str, val_samples: int = DEFAULT_VAL_SAMPLES) -> List[str]:
    """train/val split with fixed seed.

    val_samples > 0: hold out exactly N pairs for val, use all remaining pairs for train.
    val_samples == 0: legacy ratio split (_TRAIN_RATIO train / rest val).
    pred: all paired samples.
    """
    pairs = list_paired_samples(root)
    if mode == 'pred':
        return pairs

    rng = random.Random(_SPLIT_SEED)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    if val_samples > 0:
        n_val = min(val_samples, len(shuffled) - 1)
        val_part = shuffled[:n_val]
        train_part = shuffled[n_val:]
    else:
        split_at = int(len(shuffled) * _TRAIN_RATIO)
        train_part = shuffled[:split_at]
        val_part = shuffled[split_at:]

    if mode == 'train':
        return train_part
    if mode == 'val':
        return val_part
    raise ValueError(f'unknown mode: {mode}, expected train, val, or pred')


def _pad_to_min_size(stacked: Tensor, min_size: int) -> Tensor:
    _, h, w = stacked.shape
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h == 0 and pad_w == 0:
        return stacked
    return F.pad(stacked, (0, pad_w, 0, pad_h), mode='reflect')


def paired_random_crop_flip(stacked: Tensor, crop_size: int) -> Tensor:
    """Synchronized h/v flip + random crop for stacked IR/VI tensor [2, H, W]."""
    if random.random() < 0.5:
        stacked = torch.flip(stacked, dims=[2])
    if random.random() < 0.5:
        stacked = torch.flip(stacked, dims=[1])

    stacked = _pad_to_min_size(stacked, crop_size)
    _, h, w = stacked.shape
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return stacked[:, top:top + crop_size, left:left + crop_size]


def paired_center_crop(stacked: Tensor, crop_size: int) -> Tensor:
    """Synchronized center crop for validation."""
    stacked = _pad_to_min_size(stacked, crop_size)
    _, h, w = stacked.shape
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return stacked[:, top:top + crop_size, left:left + crop_size]


class IVIFDataset(Dataset):
    """Dataset layout: $root/ir and $root/vi with matching filenames."""

    def __init__(
        self,
        root: str | Path,
        mode: str = 'train',
        crop_size: int = 320,
        vi_synth: bool = True,
        vi_synth_prob: float = 0.9,
        vi_synth_regions: tuple[int, int] = (1, 3),
        synth_global_prob: float = DEFAULT_SYNTH_GLOBAL_PROB,
        val_samples: int = DEFAULT_VAL_SAMPLES,
    ):
        super().__init__()
        if vi_synth_regions[0] > vi_synth_regions[1]:
            raise ValueError(f'vi_synth_regions MIN>MAX: {vi_synth_regions}')
        self.root = Path(root)
        self.mode = mode
        self.crop_size = crop_size
        self.vi_synth = vi_synth and mode == 'train'
        self.vi_synth_prob = vi_synth_prob
        self.vi_synth_regions = vi_synth_regions
        self.synth_global_prob = synth_global_prob
        self.samples = resolve_samples(self.root, mode, val_samples=val_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        name = self.samples[index]
        ir = self.gray_read(self.root / 'ir' / name)
        vi = self.color_read(self.root / 'vi' / name)
        stacked = torch.cat([ir, vi], dim=0)

        if self.mode == 'train':
            stacked = paired_random_crop_flip(stacked, self.crop_size)
        elif self.mode == 'val':
            stacked = paired_center_crop(stacked, self.crop_size)

        ir, vi = torch.chunk(stacked, 2, dim=0)
        ir_clean, vi_clean = ir.clone(), vi.clone()
        # IR is always clean; only VI is synthesized / gated.
        h, w = ir.shape[-2], ir.shape[-1]
        vi_deg = 'normal'
        vi_mask = torch.zeros(1, h, w, dtype=vi.dtype)
        vi_type_masks = torch.zeros(len(VI_DEGRADATION_NAMES), h, w, dtype=vi.dtype)
        vi_type_masks[0] = 1.0
        if self.vi_synth:
            vi, vi_deg, vi_mask, vi_type_masks = synthesize_vi_degradation(
                vi,
                apply_prob=self.vi_synth_prob,
                min_regions=self.vi_synth_regions[0],
                max_regions=self.vi_synth_regions[1],
                global_prob=self.synth_global_prob,
            )
        return {
            'name': name,
            'ir': ir,
            'vi': vi,
            'ir_clean': ir_clean,
            'vi_clean': vi_clean,
            'vi_mask': vi_mask,
            'vi_type_masks': vi_type_masks,
            'vi_deg': vi_deg,
        }

    @staticmethod
    def color_read(img_path: str | Path) -> Tensor:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = image_to_tensor(ycrcb[:, :, 0]).float() / 255
        return y

    @staticmethod
    def gray_read(img_path: str | Path) -> Tensor:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        return image_to_tensor(img).float() / 255

    @staticmethod
    def collate_fn(data: List[dict]) -> dict:
        keys = data[0].keys()
        merged = {}
        for key in keys:
            items = [d[key] for d in data]
            merged[key] = items if isinstance(items[0], str) else torch.stack(items)
        return merged

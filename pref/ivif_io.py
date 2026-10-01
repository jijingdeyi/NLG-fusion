"""Minimal IR/VI pair listing for preference-bench builds."""

from __future__ import annotations

from pathlib import Path

_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.bmp', '.png'}


def list_paired_samples(root: Path | str) -> list[str]:
    """Return sorted filenames present in both ``ir/`` and ``vi/``."""
    root = Path(root)
    ir_dir, vi_dir = root / 'ir', root / 'vi'
    if not ir_dir.is_dir() or not vi_dir.is_dir():
        raise FileNotFoundError(f'expected directories: {ir_dir} and {vi_dir}')
    ir_names = {p.name for p in ir_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES}
    vi_names = {p.name for p in vi_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES}
    pairs = sorted(ir_names & vi_names)
    if not pairs:
        raise FileNotFoundError(f'no paired images under {ir_dir} and {vi_dir}')
    return pairs

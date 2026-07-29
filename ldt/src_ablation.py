"""SRC modulation ablation for Table tab:src-ablation."""

from __future__ import annotations

from modules.ldt import SRC_MODES

PAPER_SRC_LABEL = {
    'global_film': 'Global FiLM',
    'residual_film': 'Residual FiLM',
    'src': 'SRC (proposed)',
}


def validate_src_mode(name: str) -> str:
    if name not in SRC_MODES:
        raise ValueError(f'Unknown src_mode="{name}". Choose from: {list(SRC_MODES)}')
    return name

"""Experiment run naming and config snapshot for training."""

from __future__ import annotations

import re
from argparse import Namespace
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

from ldt.defaults import (
    DEFAULT_DIM,
    DEFAULT_F_WEIGHT,
    DEFAULT_LR,
    DEFAULT_LR_SCHED_PATIENCE,
    DEFAULT_PRIOR_HF_ALPHA,
)


def sanitize_run_name(name: str, max_len: int = 120) -> str:
    name = name.strip().replace(' ', '_')
    name = re.sub(r'[^\w.\-]+', '-', name)
    name = re.sub(r'-+', '-', name).strip('-')
    if not name:
        raise ValueError('run_name is empty after sanitization')
    return name[:max_len]


def build_run_name(args: Namespace) -> str:
    if getattr(args, 'run_name', None):
        return sanitize_run_name(args.run_name)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    return sanitize_run_name(f'fus_{ts}')


def _read_ckpt_train_args(ckpt_path: str | Path):
    import torch

    path = Path(ckpt_path)
    if not path.is_file():
        return None
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return None
    hp = ckpt.get('hyper_parameters') or {}
    return hp.get('args')


def _read_run_config(ckpt_path: str | Path) -> dict:
    cfg = Path(ckpt_path).parent / 'run_config.yaml'
    if not cfg.is_file():
        return {}
    return yaml.safe_load(cfg.read_text(encoding='utf-8')) or {}


def merge_inference_args_from_checkpoint(ckpt_path: str | Path, cli: Namespace) -> Namespace:
    args = deepcopy(cli)
    train_args = _read_ckpt_train_args(ckpt_path)
    cfg = _read_run_config(ckpt_path)

    def _get(name, default):
        if train_args is not None and hasattr(train_args, name):
            return getattr(train_args, name)
        if name in cfg:
            return cfg[name]
        return default

    args.dim = int(_get('dim', getattr(cli, 'dim', DEFAULT_DIM)))
    args.lr = float(_get('lr', getattr(cli, 'lr', DEFAULT_LR)))
    args.f_weight = list(_get('f_weight', getattr(cli, 'f_weight', list(DEFAULT_F_WEIGHT))))
    args.prior_hf_alpha = float(
        _get('prior_hf_alpha', getattr(cli, 'prior_hf_alpha', DEFAULT_PRIOR_HF_ALPHA)),
    )
    args.lr_sched_patience = int(
        _get('lr_sched_patience', getattr(cli, 'lr_sched_patience', DEFAULT_LR_SCHED_PATIENCE)),
    )
    args.ckpt = str(ckpt_path)
    return args


def snapshot_run_config(args: Namespace, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / 'run_config.yaml'
    payload = {k: getattr(args, k) for k in sorted(vars(args))}
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding='utf-8')
    return path

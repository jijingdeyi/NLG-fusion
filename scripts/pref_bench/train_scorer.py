#!/usr/bin/env python3
"""Train SharedScorer with R-weighted preference CE (score.md §4.2–§4.4).

  CUDA_VISIBLE_DEVICES=3 python scripts/pref_bench/train_scorer.py \\
      --config /data/ykx/pref_bench/config.yaml --gpu 3
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path


def _early_pin_gpu() -> None:
    gpu = '3'
    for i, a in enumerate(sys.argv):
        if a == '--gpu' and i + 1 < len(sys.argv):
            gpu = sys.argv[i + 1]
            break
        if a.startswith('--gpu='):
            gpu = a.split('=', 1)[1]
            break
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu


_early_pin_gpu()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pref.io import load_yaml
from pref.train import train_scorer


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--gpu', default='3')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--delta', type=float, default=0.25)
    ap.add_argument('--sigma0', type=float, default=0.1)
    ap.add_argument('--eta', type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = 'cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    if not __import__('torch').cuda.is_available():
        device = 'cpu'

    report = train_scorer(
        out_root=cfg['out_root'],
        cfg_path=args.config,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        size=args.size,
        delta=args.delta,
        sigma0=args.sigma0,
        eta=args.eta,
        seed=int(cfg.get('seed', 443)),
    )
    print('best:', report['best'])


if __name__ == '__main__':
    main()

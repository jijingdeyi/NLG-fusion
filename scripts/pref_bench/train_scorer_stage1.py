#!/usr/bin/env python3
"""Train source-input / leave-one-method-out ablations for Stage-1 scorer.

  # source ablation (3 jobs; full IR+VIS+F already in scorer_stage1_best.pt)
  CUDA_VISIBLE_DEVICES=0 python scripts/pref_bench/train_scorer_stage1.py \\
      --config /data/ykx/pref_bench/config.yaml --gpu 0 --input-mode f_only

  # leave-one-method-out
  CUDA_VISIBLE_DEVICES=1 python scripts/pref_bench/train_scorer_stage1.py \\
      --config /data/ykx/pref_bench/config.yaml --gpu 1 --holdout-method method_a
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path


def _early_pin_gpu() -> None:
    gpu = '0'
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
from pref.train_stage1 import train_scorer_stage1


def main() -> None:
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--freeze-epochs', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr-head', type=float, default=1e-4)
    ap.add_argument('--lr-backbone', type=float, default=1e-5)
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--control-downweight', type=float, default=0.35)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument(
        '--input-mode', default='ir_vis_f',
        choices=['ir_vis_f', 'f_only', 'ir_f', 'vis_f'],
    )
    ap.add_argument('--holdout-method', default=None,
                    help='exclude train+val pairs involving this method '
                         '(and its derivatives, e.g. deg_* for method_a)')
    ap.add_argument(
        '--matched-train-size', type=int, default=None,
        help='random train subsample size (size-matched control; methods kept)',
    )
    ap.add_argument('--tag', default=None, help='checkpoint name tag override')
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    import torch
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    report = train_scorer_stage1(
        out_root=cfg['out_root'],
        cfg_path=args.config,
        device=device,
        epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        size=args.size,
        tau=args.tau,
        control_downweight=args.control_downweight,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
        seed=int(cfg.get('seed', 443)),
        input_mode=args.input_mode,
        holdout_method=args.holdout_method,
        matched_train_size=args.matched_train_size,
        run_tag=args.tag,
        refresh_supervision=False,
    )
    print('best:', report['best'])
    print('ckpt:', report['best_ckpt'])


if __name__ == '__main__':
    main()

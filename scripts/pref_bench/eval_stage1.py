#!/usr/bin/env python3
"""Stage-1 verification on frozen best checkpoint (no hyperparam tuning).

  CUDA_VISIBLE_DEVICES=4 python scripts/pref_bench/eval_stage1.py \\
      --config /data/ykx/pref_bench/config.yaml --gpu 4

Reports test BCE / soft / decisive accuracy, preference-strength buckets,
traditional-metric baselines under the same pairwise protocol, and
Δs vs q_A scatter plots. Does NOT train Stage-2.
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path


def _early_pin_gpu() -> None:
    gpu = '4'
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

from pref.eval_stage1 import format_markdown_table, run_stage1_eval
from pref.io import load_yaml


def main() -> None:
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--gpu', default='4')
    ap.add_argument('--ckpt', type=Path, default=None)
    ap.add_argument('--batch-size', type=int, default=16)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    import torch
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    report = run_stage1_eval(
        out_root=cfg['out_root'],
        ckpt_path=args.ckpt,
        device=device,
        batch_size=args.batch_size,
    )

    print(f'ckpt: {report["ckpt"]} (epoch {report["ckpt_epoch"]})')
    print('--- splits ---')
    for split, s in report['splits'].items():
        print(
            f'  {split:5s}  n={s["n"]:4d}  loss={s["loss"]:.4f}  '
            f'soft={s["acc_soft"]:.3f}  dec={s["acc_decisive"]:.3f}  '
            f'(decisive {s["n_decisive"]})'
        )
    print('--- test preference-strength buckets ---')
    for b in report['test_buckets']:
        print(
            f'  {b["bucket"]:12s}  n={b["n"]:3d}  soft={b["acc_soft"]:.3f}  '
            f'dec={b["acc_decisive"]:.3f}'
        )
    print('--- decisive-pair baselines (test) ---')
    print(format_markdown_table(report['baselines_decisive']))
    sp = report.get('spearman_delta_s_vs_q_test')
    if sp:
        print(f'Spearman Δs vs q (test): ρ={sp["rho"]:.3f}  p={sp["p"]}')
    print(f'edge consistency (test): {report["edge_consistency_test"]["overall"]:.3f}')
    print(f'report: {report["report_path"]}')
    print(f'plot:   {report["plots"]["test"]}')


if __name__ == '__main__':
    main()

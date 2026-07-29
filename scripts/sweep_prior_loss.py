#!/usr/bin/env python3
"""Sweep prior HF blend α (no train) then int/tex weights (short train).

Usage:
  python scripts/sweep_prior_loss.py --data /data/ykx/LLVIP/train

Grids are fixed in this file on purpose — no mode switches.
"""

from __future__ import annotations

import csv
import logging
import sys
import warnings
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ldt.dataset import IVIFDataset
from ldt.defaults import (
    DEFAULT_CROP_SIZE,
    DEFAULT_DIM,
    DEFAULT_LR,
    DEFAULT_LR_SCHED_PATIENCE,
    DEFAULT_VAL_SAMPLES,
)
from ldt.fusion_metrics import VAL_METRIC_RANGES, compute_fusion_metrics
from ldt.fusion_prior import build_fusion_prior
from ldt.ldt_fusion import LDTFusion


def _norm_metric(name: str, value: float) -> float:
    lo, hi = VAL_METRIC_RANGES[name]
    if hi <= lo:
        return 0.0
    return float(max(0.0, min(1.0, (value - lo) / (hi - lo))))


def balance_mi_qabf(mi: float, qabf: float) -> float:
    """Prefer neither pure-max (α=0) nor pure-HF; used only to pick α for phase 2."""
    return 0.5 * (_norm_metric('mi', mi) + _norm_metric('qabf', qabf))


def pick_alpha(rows: list[dict]) -> dict:
    """Pick α for phase-2 from the MI–Qabf Pareto compromise.

    Pure ``α=0`` (classical max) usually wins raw ``score`` via MI; it is kept in
    the table for reference but excluded from auto-pick so phase-2 actually tests
    a textured residual base.
    """
    candidates = [r for r in rows if r['alpha'] > 0.0]
    if not candidates:
        candidates = rows
    mi_lo = min(r['mi'] for r in candidates)
    mi_hi = max(r['mi'] for r in candidates)
    q_lo = min(r['qabf'] for r in candidates)
    q_hi = max(r['qabf'] for r in candidates)

    def utopia(r: dict) -> float:
        nm = (r['mi'] - mi_lo) / (mi_hi - mi_lo + 1e-8)
        nq = (r['qabf'] - q_lo) / (q_hi - q_lo + 1e-8)
        return ((1.0 - nm) ** 2 + (1.0 - nq) ** 2) ** 0.5

    for r in candidates:
        r['utopia'] = utopia(r)
        r['balance'] = balance_mi_qabf(r['mi'], r['qabf'])
    return min(candidates, key=lambda r: r['utopia'])

# Fixed grids — edit here if you need different points.
ALPHA_GRID = (0.0, 0.2, 0.3, 0.5, 0.7, 1.0)
TEX_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)  # int_w = 1 - tex_w, ssim = 0
SHORT_EPOCHS = 8
SHORT_PATIENCE = 3
# Cap train work so phase-2 stays a screen of hours, not a full LLVIP epoch × 5.
LIMIT_TRAIN_BATCHES = 200
LIMIT_VAL_BATCHES = 0  # 0 = full val loader


def _metrics_batch(ir: torch.Tensor, vi: torch.Tensor, fused: torch.Tensor) -> dict[str, float]:
    totals = {k: 0.0 for k in ('mi', 'qabf', 'scd', 'vif', 'ssim', 'score')}
    n = ir.shape[0]
    for i in range(n):
        ir_np = (ir[i, 0].detach().cpu().numpy() * 255.0).astype('float32')
        vi_np = (vi[i, 0].detach().cpu().numpy() * 255.0).astype('float32')
        f_np = (fused[i, 0].detach().cpu().numpy() * 255.0).astype('float32')
        m = compute_fusion_metrics(ir_np, vi_np, f_np)
        for k in totals:
            totals[k] += m[k]
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def eval_prior_alpha(loader: DataLoader, alpha: float, device: torch.device) -> dict[str, float]:
    acc = {k: 0.0 for k in ('mi', 'qabf', 'scd', 'vif', 'ssim', 'score')}
    count = 0
    for batch in loader:
        ir = batch['ir'].to(device)
        vi = batch['vi'].to(device)
        prior = build_fusion_prior(ir, vi, hf_alpha=alpha)
        m = _metrics_batch(ir, vi, prior)
        b = ir.shape[0]
        for k in acc:
            acc[k] += m[k] * b
        count += b
    return {k: acc[k] / max(count, 1) for k in acc}


def _print_table(title: str, rows: list[dict], keys: list[str]) -> None:
    print(f'\n=== {title} ===')
    header = '  '.join(f'{k:>8}' for k in keys)
    print(header)
    for row in rows:
        print('  '.join(f'{row[k]:8.4f}' if isinstance(row[k], float) else f'{row[k]:>8}' for k in keys))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def phase1_prior(loader: DataLoader, device: torch.device, out_dir: Path) -> float:
    rows = []
    for alpha in ALPHA_GRID:
        m = eval_prior_alpha(loader, alpha, device)
        row = {'alpha': alpha, **m}
        rows.append(row)
        print(
            f'[prior] α={alpha:.1f}  score={m["score"]:.4f}  '
            f'mi={m["mi"]:.3f}  qabf={m["qabf"]:.3f}',
        )

    best = pick_alpha(rows)
    for r in rows:
        r.setdefault('balance', balance_mi_qabf(r['mi'], r['qabf']))
        r.setdefault('utopia', float('nan'))

    _print_table(
        'Phase-1 prior only (u=m, no train)',
        rows,
        ['alpha', 'utopia', 'score', 'mi', 'qabf', 'vif', 'ssim'],
    )
    _write_csv(out_dir / 'phase1_prior.csv', rows)

    print(
        f'\n[prior] pick α={best["alpha"]:.1f} by utopia distance={best["utopia"]:.4f} '
        f'(α=0 excluded from auto-pick; see table for full Pareto)',
    )
    return float(best['alpha'])


def phase2_weights(
    alpha: float,
    args: Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    out_dir: Path,
) -> None:
    rows = []
    for tex_w in TEX_GRID:
        int_w = 1.0 - tex_w
        run_name = f'sweep_a{alpha:.1f}_t{tex_w:.1f}'
        run_dir = out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        ns = Namespace(
            dim=DEFAULT_DIM,
            lr=DEFAULT_LR,
            f_weight=[int_w, 0.0, tex_w],
            prior_hf_alpha=alpha,
            lr_sched_patience=DEFAULT_LR_SCHED_PATIENCE,
        )
        print(f'\n[short] α={alpha:.1f}  f_weight=({int_w:.1f}, 0, {tex_w:.1f})  epochs={args.epochs}')
        pl.seed_everything(args.seed)
        model = LDTFusion(ns)
        ckpt_cb = ModelCheckpoint(
            dirpath=str(run_dir),
            filename=f'{run_name}-best-e{{epoch:03d}}',
            monitor='val/score',
            mode='max',
            save_top_k=1,
        )
        trainer = pl.Trainer(
            accelerator='cpu' if args.no_cuda else 'gpu',
            devices=1,
            logger=False,
            callbacks=[
                ckpt_cb,
                EarlyStopping(
                    monitor='val/score', mode='max',
                    patience=SHORT_PATIENCE, check_finite=True,
                ),
            ],
            max_epochs=args.epochs,
            limit_train_batches=LIMIT_TRAIN_BATCHES,
            limit_val_batches=None if LIMIT_VAL_BATCHES <= 0 else LIMIT_VAL_BATCHES,
            enable_model_summary=False,
            enable_progress_bar=True,
            log_every_n_steps=50,
        )
        trainer.fit(model, train_loader, val_loader)
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith('val/')}
        row = {
            'alpha': alpha,
            'int_w': int_w,
            'tex_w': tex_w,
            'score': metrics.get('val/score', float('nan')),
            'mi': metrics.get('val/mi', float('nan')),
            'qabf': metrics.get('val/qabf', float('nan')),
            'vif': metrics.get('val/vif', float('nan')),
            'ssim': metrics.get('val/ssim', float('nan')),
            'best_ckpt': str(ckpt_cb.best_model_path or ''),
        }
        rows.append(row)
        print(
            f'[short] done  score={row["score"]:.4f}  mi={row["mi"]:.3f}  qabf={row["qabf"]:.3f}',
        )

    _print_table(
        f'Phase-2 short train (α={alpha:.1f} fixed)',
        rows,
        ['int_w', 'tex_w', 'score', 'mi', 'qabf', 'vif'],
    )
    _write_csv(out_dir / 'phase2_weights.csv', rows)
    best = max(rows, key=lambda r: r['score'] if r['score'] == r['score'] else -1.0)
    print(
        f'\n[recommend] prior_hf_alpha={best["alpha"]:.1f}  '
        f'f_weight {best["int_w"]:.1f} 0 {best["tex_w"]:.1f}  '
        f'(short-train score={best["score"]:.4f})',
    )
    print(
        f'  full train: python scripts/train.py --data {args.data} '
        f'--prior_hf_alpha {best["alpha"]:.1f} '
        f'--f_weight {best["int_w"]:.1f} 0 {best["tex_w"]:.1f}',
    )


def main():
    parser = ArgumentParser(description='Sweep prior α then int/tex weights.')
    parser.add_argument('--data', type=str, default='/data/ykx/LLVIP/train')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--val_bs', type=int, default=4)
    parser.add_argument('--val_samples', type=int, default=DEFAULT_VAL_SAMPLES)
    parser.add_argument('--crop_size', type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument('--epochs', type=int, default=SHORT_EPOCHS, help='Short-train epochs (phase 2)')
    parser.add_argument('--seed', type=int, default=443)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
    logging.getLogger('lightning.pytorch').setLevel(logging.WARNING)

    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    stamp = datetime.now().strftime('%Y%m%d_%H%M')
    out_dir = Path(args.log_dir) / 'sweeps' / f'prior_loss_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[sweep] results -> {out_dir.resolve()}')
    print(f'[sweep] device={device}  val_samples={args.val_samples}')

    train_loader = DataLoader(
        IVIFDataset(
            args.data, 'train', crop_size=args.crop_size,
            vi_synth=False, val_samples=args.val_samples,
        ),
        batch_size=args.bs, shuffle=True,
        collate_fn=IVIFDataset.collate_fn, num_workers=4, pin_memory=device.type == 'cuda',
    )
    val_loader = DataLoader(
        IVIFDataset(
            args.data, 'val', crop_size=args.crop_size,
            vi_synth=False, val_samples=args.val_samples,
        ),
        batch_size=args.val_bs,
        collate_fn=IVIFDataset.collate_fn, num_workers=4, pin_memory=device.type == 'cuda',
    )

    alpha = phase1_prior(val_loader, device, out_dir)
    phase2_weights(alpha, args, train_loader, val_loader, out_dir)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()

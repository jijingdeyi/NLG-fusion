#!/usr/bin/env python3
"""Sweep teacher backbone prior α then int/tex weights using max-min val/score.

Usage:
  python scripts/sweep_prior_loss.py --data /data/ykx/LLVIP/train

Grids are fixed in this file on purpose — no mode switches.
Selection criterion: COMPETITIVE max-min ``val/score`` (see ldt/fusion_metrics.py).
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
    DEFAULT_SCORE_RECIPE,
    DEFAULT_VAL_SAMPLES,
)
from ldt.fusion_metrics import (
    ALL_METRIC_KEYS,
    COMPETITIVE_RANGES,
    SCORE_RECIPES,
    score_from_metrics,
    compute_fusion_metrics,
)
from ldt.fusion_prior import build_fusion_prior
from ldt.ldt_fusion import LDTFusion

# Fixed grids — edit here if you need different points.
ALPHA_GRID = (0.0, 0.2, 0.3, 0.5, 0.7, 1.0)
TEX_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)  # int_w = 1 - tex_w, ssim = 0
SHORT_EPOCHS = 8
SHORT_PATIENCE = 3
LIMIT_TRAIN_BATCHES = 200
LIMIT_VAL_BATCHES = 0  # 0 = full val loader


def _raw_metrics_batch(ir: torch.Tensor, vi: torch.Tensor, fused: torch.Tensor) -> dict[str, float]:
    """Average raw metrics only (no per-image score)."""
    totals = {k: 0.0 for k in ALL_METRIC_KEYS}
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
def eval_prior_alpha(
    loader: DataLoader, alpha: float, device: torch.device, recipe: str,
) -> dict[str, float]:
    acc = {k: 0.0 for k in ALL_METRIC_KEYS}
    count = 0
    for batch in loader:
        ir = batch['ir'].to(device)
        vi = batch['vi'].to(device)
        prior = build_fusion_prior(ir, vi, hf_alpha=alpha)
        m = _raw_metrics_batch(ir, vi, prior)
        b = ir.shape[0]
        for k in acc:
            acc[k] += m[k] * b
        count += b
    means = {k: acc[k] / max(count, 1) for k in acc}
    score, details = score_from_metrics(means, recipe=recipe)
    return {
        **means,
        'score': score,
        'worst_metric': str(details['worst_metric']),
        'worst_attainment': float(details['worst_attainment']),
    }


def _print_table(title: str, rows: list[dict], keys: list[str]) -> None:
    print(f'\n=== {title} ===')
    header = '  '.join(f'{k:>10}' for k in keys)
    print(header)
    for row in rows:
        cells = []
        for k in keys:
            v = row[k]
            if isinstance(v, float):
                cells.append(f'{v:10.4f}')
            else:
                cells.append(f'{str(v):>10}')
        print('  '.join(cells))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def phase1_prior(loader: DataLoader, device: torch.device, out_dir: Path, recipe: str) -> float:
    rows = []
    for alpha in ALPHA_GRID:
        m = eval_prior_alpha(loader, alpha, device, recipe=recipe)
        row = {'alpha': alpha, **m}
        rows.append(row)
        print(
            f'[prior] α={alpha:.1f}  score={m["score"]:.4f}  '
            f'worst={m["worst_metric"]}({m["worst_attainment"]:+.3f})  '
            f'mi={m["mi"]:.3f}  qabf={m["qabf"]:.3f}',
        )

    _print_table(
        f'Phase-1 prior only (u=m) — recipe={recipe} keys={SCORE_RECIPES[recipe]}',
        rows,
        ['alpha', 'score', 'worst_metric', 'mi', 'qabf', 'scd', 'vif', 'ssim', 'sf'],
    )
    _write_csv(out_dir / 'phase1_prior.csv', rows)

    best = max(rows, key=lambda r: r['score'])
    print(
        f'\n[prior] pick α={best["alpha"]:.1f} by max-min score={best["score"]:.4f} '
        f'(worst={best["worst_metric"]}, recipe={recipe})',
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
            score_recipe=args.score_recipe,
            lr_sched_patience=DEFAULT_LR_SCHED_PATIENCE,
        )
        print(
            f'\n[short] α={alpha:.1f}  f_weight=({int_w:.1f}, 0, {tex_w:.1f})  '
            f'recipe={args.score_recipe}  epochs={args.epochs}',
        )
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
        means = {k: metrics.get(f'val/{k}', float('nan')) for k in ALL_METRIC_KEYS}
        score = metrics.get('val/score', float('nan'))
        if score != score:
            score, details = score_from_metrics(means, recipe=args.score_recipe)
        else:
            _, details = score_from_metrics(means, recipe=args.score_recipe)
        row = {
            'alpha': alpha,
            'int_w': int_w,
            'tex_w': tex_w,
            'score': score,
            **means,
            'worst_metric': details['worst_metric'],
            'worst_attainment': details['worst_attainment'],
            'best_ckpt': str(ckpt_cb.best_model_path or ''),
        }
        rows.append(row)
        print(
            f'[short] done  score={row["score"]:.4f}  '
            f'worst={row["worst_metric"]}  mi={row["mi"]:.3f}  qabf={row["qabf"]:.3f}',
        )

    _print_table(
        f'Phase-2 short train (α={alpha:.1f}, recipe={args.score_recipe})',
        rows,
        ['int_w', 'tex_w', 'score', 'worst_metric', 'mi', 'qabf', 'sf', 'vif'],
    )
    _write_csv(out_dir / 'phase2_weights.csv', rows)
    best = max(rows, key=lambda r: r['score'] if r['score'] == r['score'] else -1.0)
    print(
        f'\n[recommend] prior_hf_alpha={best["alpha"]:.1f}  '
        f'f_weight {best["int_w"]:.1f} 0 {best["tex_w"]:.1f}  '
        f'score_recipe={args.score_recipe}  '
        f'(short-train score={best["score"]:.4f}, worst={best["worst_metric"]})',
    )
    print(
        f'  full train: python scripts/train.py --data {args.data} '
        f'--prior_hf_alpha {best["alpha"]:.1f} '
        f'--f_weight {best["int_w"]:.1f} 0 {best["tex_w"]:.1f} '
        f'--score_recipe {args.score_recipe}',
    )


def main():
    parser = ArgumentParser(description='Sweep teacher prior α / loss weights via max-min score.')
    parser.add_argument('--data', type=str, default='/data/ykx/LLVIP/train')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--val_bs', type=int, default=4)
    parser.add_argument('--val_samples', type=int, default=DEFAULT_VAL_SAMPLES)
    parser.add_argument('--crop_size', type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument('--epochs', type=int, default=SHORT_EPOCHS, help='Short-train epochs (phase 2)')
    parser.add_argument('--seed', type=int, default=443)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument(
        '--score_recipe', type=str, default=DEFAULT_SCORE_RECIPE,
        choices=sorted(SCORE_RECIPES),
        help='Max-min score recipe A–F',
    )
    args = parser.parse_args()

    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
    logging.getLogger('lightning.pytorch').setLevel(logging.WARNING)

    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    stamp = datetime.now().strftime('%Y%m%d_%H%M')
    out_dir = Path(args.log_dir) / 'sweeps' / f'prior_loss_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[sweep] results -> {out_dir.resolve()}')
    print(f'[sweep] device={device}  val_samples={args.val_samples}')
    print(f'[sweep] score_recipe={args.score_recipe}  keys={SCORE_RECIPES[args.score_recipe]}')
    print('[sweep] COMPETITIVE_RANGES (LLVIP median→max):')
    for k in SCORE_RECIPES[args.score_recipe]:
        lo, hi = COMPETITIVE_RANGES[k]
        print(f'         {k}: floor={lo:.4f}  target={hi:.4f}')

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

    alpha = phase1_prior(val_loader, device, out_dir, recipe=args.score_recipe)
    phase2_weights(alpha, args, train_loader, val_loader, out_dir)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()

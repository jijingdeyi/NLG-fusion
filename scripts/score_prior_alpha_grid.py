#!/usr/bin/env python3
"""§8.2: score prior pseudo-targets m(α) with recipes A–F.

No training. Builds m(α)=(1-α)·max + α·max_hf on the val split, averages
metrics, then scores with each SCORE_RECIPE. Optionally saves a visual strip
for one sample.

Usage:
  python scripts/score_prior_alpha_grid.py --data /data/ykx/LLVIP/train
  python scripts/score_prior_alpha_grid.py --data /data/ykx/LLVIP/train --out runs/score_prior_alpha
"""

from __future__ import annotations

import csv
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ldt.dataset import IVIFDataset
from ldt.defaults import DEFAULT_CROP_SIZE, DEFAULT_VAL_SAMPLES
from ldt.fusion_metrics import (
    ALL_METRIC_KEYS,
    COMPETITIVE_RANGES,
    SCORE_RECIPES,
    compute_fusion_metrics,
    score_from_metrics,
)
from ldt.fusion_prior import build_fusion_prior

ALPHA_GRID = (0.0, 0.2, 0.3, 0.5, 0.7, 1.0)


def _raw_metrics_batch(ir: torch.Tensor, vi: torch.Tensor, fused: torch.Tensor) -> dict[str, float]:
    totals = {k: 0.0 for k in ALL_METRIC_KEYS}
    n = ir.shape[0]
    for i in range(n):
        ir_np = (ir[i, 0].detach().cpu().numpy() * 255.0).astype('float32')
        vi_np = (vi[i, 0].detach().cpu().numpy() * 255.0).astype('float32')
        f_np = (fused[i, 0].detach().cpu().numpy() * 255.0).astype('float32')
        m = compute_fusion_metrics(ir_np, vi_np, f_np)
        for k in totals:
            totals[k] += float(m[k])
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def eval_means(
    loader: DataLoader, alpha: float, device: torch.device,
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
    return {k: acc[k] / max(count, 1) for k in acc}


@torch.no_grad()
def save_qual_strip(
    dataset: IVIFDataset,
    out_path: Path,
    device: torch.device,
    sample_index: int = 0,
) -> str:
    """Save IR | VI | m(α)… strip for one val sample. Returns sample name."""
    import cv2

    sample = dataset[sample_index]
    name = str(sample.get('name', sample.get('id', f'idx{sample_index}')))
    ir = sample['ir'].unsqueeze(0).to(device)
    vi = sample['vi'].unsqueeze(0).to(device)

    def to_u8(t: torch.Tensor) -> np.ndarray:
        x = (t[0, 0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return x

    panels = [to_u8(ir), to_u8(vi)]
    labels = ['IR', 'VI']
    for a in ALPHA_GRID:
        panels.append(to_u8(build_fusion_prior(ir, vi, hf_alpha=a)))
        labels.append(f'a={a:.1f}')

    h = panels[0].shape[0]
    # resize width if needed (same crop → same size)
    gap = 4
    bar = 28
    widths = [p.shape[1] for p in panels]
    canvas_w = sum(widths) + gap * (len(panels) - 1)
    canvas = np.full((h + bar, canvas_w), 255, dtype=np.uint8)
    x = 0
    for p, lab, w in zip(panels, labels, widths):
        canvas[bar:bar + h, x:x + w] = p
        cv2.putText(
            canvas, lab, (x + 4, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA,
        )
        x += w + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return name


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = ArgumentParser(description=__doc__)
    p.add_argument('--data', type=str, default='/data/ykx/LLVIP/train')
    p.add_argument('--out', type=Path, default=Path('runs/score_prior_alpha'))
    p.add_argument('--val_samples', type=int, default=DEFAULT_VAL_SAMPLES)
    p.add_argument('--crop_size', type=int, default=DEFAULT_CROP_SIZE)
    p.add_argument('--val_bs', type=int, default=4)
    p.add_argument('--qual_index', type=int, default=0, help='Val sample index for strip')
    p.add_argument('--no_cuda', action='store_true')
    args = p.parse_args()

    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    args.out.mkdir(parents=True, exist_ok=True)

    val_ds = IVIFDataset(
        args.data, 'val', crop_size=args.crop_size,
        vi_synth=False, val_samples=args.val_samples,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_bs,
        collate_fn=IVIFDataset.collate_fn, num_workers=4,
        pin_memory=device.type == 'cuda',
    )
    print(f'[8.2] device={device}  val_n={len(val_ds)}  out={args.out.resolve()}')

    metric_rows: list[dict] = []
    for alpha in ALPHA_GRID:
        print(f'[8.2] eval α={alpha:.1f} ...')
        means = eval_means(val_loader, alpha, device)
        metric_rows.append({'alpha': alpha, **means})
        print(
            '         '
            + '  '.join(f'{k}={means[k]:.4f}' for k in ALL_METRIC_KEYS),
        )
    _write_csv(args.out / 'metrics_by_alpha.csv', metric_rows)

    score_rows: list[dict] = []
    # wide table: alpha × recipe scores
    wide: list[dict] = []
    for row in metric_rows:
        alpha = float(row['alpha'])
        metrics = {k: float(row[k]) for k in ALL_METRIC_KEYS}
        wide_row: dict = {
            'alpha': alpha,
            **{k: metrics[k] for k in ALL_METRIC_KEYS},
        }
        for recipe in SCORE_RECIPES:
            s, d = score_from_metrics(metrics, recipe=recipe)
            score_rows.append({
                'alpha': alpha,
                'recipe': recipe,
                'score': s,
                'worst_metric': d['worst_metric'],
                'worst_attainment': d['worst_attainment'],
                'raw_score': d['raw_score'],
                **{k: metrics[k] for k in ALL_METRIC_KEYS},
            })
            wide_row[f'S_{recipe}'] = s
            wide_row[f'worst_{recipe}'] = d['worst_metric']
        wide.append(wide_row)

    _write_csv(args.out / 'scores_long.csv', score_rows)
    _write_csv(args.out / 'scores_wide.csv', wide)

    # ranking per recipe
    rank_lines = ['# §8.2 prior m(α) scores', '']
    rank_lines.append(f'data=`{args.data}`  val_samples={args.val_samples}  crop={args.crop_size}')
    rank_lines.append('')
    rank_lines.append('## Raw metrics (val mean)')
    rank_lines.append('')
    rank_lines.append('| α | MI | Qabf | SF | SCD | VIF | SSIM |')
    rank_lines.append('|--:|---:|-----:|---:|----:|----:|-----:|')
    for row in metric_rows:
        rank_lines.append(
            f'| {row["alpha"]:.1f} | {row["mi"]:.3f} | {row["qabf"]:.3f} | '
            f'{row["sf"]:.2f} | {row["scd"]:.3f} | {row["vif"]:.3f} | {row["ssim"]:.3f} |',
        )

    rank_lines.append('')
    rank_lines.append('## S by recipe (higher better)')
    rank_lines.append('')
    hdr = '| α | ' + ' | '.join(f'S_{r}' for r in SCORE_RECIPES) + ' |'
    sep = '|--:|' + '|'.join(['---:'] * len(SCORE_RECIPES)) + '|'
    rank_lines.append(hdr)
    rank_lines.append(sep)
    for wrow in wide:
        cells = ' | '.join(f'{wrow[f"S_{r}"]:.4f}' for r in SCORE_RECIPES)
        rank_lines.append(f'| {wrow["alpha"]:.1f} | {cells} |')

    rank_lines.append('')
    rank_lines.append('## Best α per recipe')
    rank_lines.append('')
    rank_lines.append('| recipe | best α | S | worst |')
    rank_lines.append('|--------|-------:|--:|-------|')
    for recipe in SCORE_RECIPES:
        sub = [r for r in score_rows if r['recipe'] == recipe]
        best = max(sub, key=lambda r: r['score'])
        rank_lines.append(
            f'| {recipe} | {best["alpha"]:.1f} | {best["score"]:.4f} | {best["worst_metric"]} |',
        )

    # direction check vs alpha=0
    rank_lines.append('')
    rank_lines.append('## Direction check: is S(α=0) the maximum?')
    rank_lines.append('')
    rank_lines.append('| recipe | S(0) max? | argmax α | pass? |')
    rank_lines.append('|--------|-----------|----------|-------|')
    for recipe in SCORE_RECIPES:
        sub = sorted(
            [r for r in score_rows if r['recipe'] == recipe],
            key=lambda r: r['alpha'],
        )
        s0 = sub[0]['score']
        best = max(sub, key=lambda r: r['score'])
        is_max0 = abs(best['score'] - s0) < 1e-12 and best['alpha'] == 0.0
        # also fail if alpha=0 uniquely highest
        is_max0 = best['alpha'] == 0.0
        mid = [r for r in sub if 0.2 <= r['alpha'] <= 0.7]
        mid_better = any(r['score'] > s0 + 1e-6 for r in mid)
        passed = (not is_max0) and mid_better
        rank_lines.append(
            f'| {recipe} | {"yes" if is_max0 else "no"} | {best["alpha"]:.1f} | '
            f'{"PASS" if passed else "FAIL"} |',
        )

    qual_name = save_qual_strip(
        val_ds, args.out / 'qual_strip.png', device, sample_index=args.qual_index,
    )
    rank_lines.append('')
    rank_lines.append(f'## Qualitative strip: `{qual_name}` → `qual_strip.png`')
    rank_lines.append('')
    rank_lines.append('Panels: IR | VI | m(α) for α in grid. Compare by eye to S rankings.')

    (args.out / 'report.md').write_text('\n'.join(rank_lines) + '\n', encoding='utf-8')
    print(f'[8.2] wrote {args.out / "report.md"}')
    print(f'[8.2] qual sample={qual_name}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Offline score recipes A–F on published method tables.

Reads each sheet's ``mean`` row from ``/data/ykx/metric/*.xlsx`` → competitive
ranges (median→max on that dataset) → six max-min scores → ranking tables + CSV.

Usage:
  python scripts/score_recipes_offline.py
  python scripts/score_recipes_offline.py --metric-dir /data/ykx/metric --out runs/score_recipes
"""

from __future__ import annotations

import csv
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ldt.fusion_metrics import (
    ALL_METRIC_KEYS,
    COMPETITIVE_RANGES,
    SCORE_RECIPES,
    build_competitive_ranges_from_means,
    load_method_means_from_xlsx,
    score_from_metrics,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fieldnames})


def _rank_descending(scores: list[float]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for r, i in enumerate(order, start=1):
        ranks[i] = r
    return ranks


def process_workbook(xlsx: Path, out_dir: Path) -> list[dict]:
    dataset = xlsx.stem
    methods, means = load_method_means_from_xlsx(str(xlsx), metric_keys=ALL_METRIC_KEYS)
    ranges = build_competitive_ranges_from_means(means, metric_keys=ALL_METRIC_KEYS)

    ds_dir = out_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    mean_rows = [
        {'method': m, **{k: f'{means[m][k]:.6f}' for k in ALL_METRIC_KEYS}}
        for m in methods
    ]
    _write_csv(ds_dir / 'method_means.csv', ['method', *ALL_METRIC_KEYS], mean_rows)

    range_rows = [
        {
            'metric': k,
            'floor': f'{ranges[k][0]:.6f}',
            'target': f'{ranges[k][1]:.6f}',
        }
        for k in ALL_METRIC_KEYS
    ]
    _write_csv(ds_dir / 'ranges.csv', ['metric', 'floor', 'target'], range_rows)

    all_rows: list[dict] = []
    ranking_lines = [f'# {dataset}', '']

    for recipe, keys in SCORE_RECIPES.items():
        scored: list[dict] = []
        for method in methods:
            s, details = score_from_metrics(
                means[method], recipe=recipe, ranges=ranges,
            )
            row = {
                'dataset': dataset,
                'recipe': recipe,
                'method': method,
                'score': s,
                'raw_score': float(details['raw_score']),
                'worst_metric': str(details['worst_metric']),
                'worst_attainment': float(details['worst_attainment']),
                'recipe_keys': str(details['recipe_keys']),
            }
            for k in keys:
                row[f'a_{k}'] = float(details[f'a_{k}'])
                row[k] = float(means[method][k])
            scored.append(row)

        scores = [r['score'] for r in scored]
        for r, rank in zip(scored, _rank_descending(scores)):
            r['rank'] = rank
        scored.sort(key=lambda r: r['rank'])

        fieldnames = [
            'rank', 'method', 'score', 'raw_score',
            'worst_metric', 'worst_attainment',
            *[f'a_{k}' for k in keys],
            *keys,
        ]
        csv_rows = []
        for r in scored:
            csv_rows.append({
                'rank': r['rank'],
                'method': r['method'],
                'score': f'{r["score"]:.6f}',
                'raw_score': f'{r["raw_score"]:.6f}',
                'worst_metric': r['worst_metric'],
                'worst_attainment': f'{r["worst_attainment"]:+.4f}',
                **{f'a_{k}': f'{r[f"a_{k}"]:+.4f}' for k in keys},
                **{k: f'{r[k]:.6f}' for k in keys},
            })
        _write_csv(ds_dir / f'scores_{recipe}.csv', fieldnames, csv_rows)

        ranking_lines.append(f'## Recipe {recipe} ({",".join(keys)})')
        ranking_lines.append('')
        ranking_lines.append('| rank | method | S | worst | a_worst |')
        ranking_lines.append('|-----:|--------|--:|-------|--------:|')
        for r in scored:
            ranking_lines.append(
                f'| {r["rank"]} | {r["method"]} | {r["score"]:.4f} | '
                f'{r["worst_metric"]} | {r["worst_attainment"]:+.3f} |',
            )
        ranking_lines.append('')

        # worst_metric histogram for this recipe
        hist: dict[str, int] = {}
        for r in scored:
            hist[r['worst_metric']] = hist.get(r['worst_metric'], 0) + 1
        ranking_lines.append('worst_metric counts: ' + ', '.join(
            f'{k}={hist[k]}' for k in sorted(hist, key=hist.get, reverse=True)
        ))
        ranking_lines.append('')

        all_rows.extend(scored)

    (ds_dir / 'rankings.md').write_text('\n'.join(ranking_lines) + '\n', encoding='utf-8')
    return all_rows


def write_summary(out_dir: Path, all_rows: list[dict]) -> None:
    """Cross-dataset CSV: long scores + per-dataset rank matrix."""
    if not all_rows:
        return

    # Long format (all a_* / raw metrics that appear)
    extra_keys = sorted({
        k for r in all_rows for k in r
        if k.startswith('a_') or k in ALL_METRIC_KEYS
    })
    long_fields = [
        'dataset', 'recipe', 'rank', 'method', 'score', 'raw_score',
        'worst_metric', 'worst_attainment', 'recipe_keys', *extra_keys,
    ]
    long_csv = []
    for r in all_rows:
        long_csv.append({
            'dataset': r['dataset'],
            'recipe': r['recipe'],
            'rank': r['rank'],
            'method': r['method'],
            'score': f'{r["score"]:.6f}',
            'raw_score': f'{r["raw_score"]:.6f}',
            'worst_metric': r['worst_metric'],
            'worst_attainment': f'{r["worst_attainment"]:+.4f}',
            'recipe_keys': r['recipe_keys'],
            **{
                k: (f'{r[k]:+.4f}' if k.startswith('a_') else f'{r[k]:.6f}')
                for k in extra_keys if k in r
            },
        })
    _write_csv(out_dir / 'all_scores.csv', long_fields, long_csv)

    # Rank matrix: one file per recipe, methods × datasets
    datasets = sorted({r['dataset'] for r in all_rows})
    recipes = list(SCORE_RECIPES)
    for recipe in recipes:
        subset = [r for r in all_rows if r['recipe'] == recipe]
        methods = sorted({r['method'] for r in subset})
        # method may be missing on some datasets
        lookup = {(r['dataset'], r['method']): r['rank'] for r in subset}
        fields = ['method', *[f'rank_{d}' for d in datasets], 'mean_rank']
        rows = []
        for m in methods:
            ranks = [lookup.get((d, m)) for d in datasets]
            present = [x for x in ranks if x is not None]
            mean_rank = sum(present) / len(present) if present else ''
            rows.append({
                'method': m,
                **{f'rank_{d}': (r if r is not None else '') for d, r in zip(datasets, ranks)},
                'mean_rank': f'{mean_rank:.2f}' if present else '',
            })
        rows.sort(key=lambda x: float(x['mean_rank']) if x['mean_rank'] else 1e9)
        _write_csv(out_dir / f'rank_matrix_{recipe}.csv', fields, rows)

    # LLVIP range sanity vs baked-in defaults
    llvip_ranges = out_dir / 'LLVIP' / 'ranges.csv'
    sanity = ['# LLVIP ranges vs COMPETITIVE_RANGES', '']
    if llvip_ranges.exists():
        with llvip_ranges.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = row['metric']
                floor, target = float(row['floor']), float(row['target'])
                ref = COMPETITIVE_RANGES.get(k)
                if ref is None:
                    sanity.append(f'- {k}: no default')
                    continue
                ok = abs(floor - ref[0]) < 1e-5 and abs(target - ref[1]) < 1e-5
                sanity.append(
                    f'- {k}: computed=({floor:.6f},{target:.6f}) '
                    f'default=({ref[0]:.6f},{ref[1]:.6f}) {"OK" if ok else "MISMATCH"}',
                )
    (out_dir / 'llvip_ranges_check.md').write_text('\n'.join(sanity) + '\n', encoding='utf-8')


def main() -> None:
    p = ArgumentParser(description=__doc__)
    p.add_argument('--metric-dir', type=Path, default=Path('/data/ykx/metric'))
    p.add_argument('--out', type=Path, default=Path('runs/score_recipes_offline'))
    p.add_argument(
        '--datasets', nargs='*', default=None,
        help='Optional stems to include (e.g. LLVIP FMB). Default: all *.xlsx',
    )
    args = p.parse_args()

    xlsxs = sorted(args.metric_dir.glob('*.xlsx'))
    if args.datasets:
        want = {s.lower() for s in args.datasets}
        xlsxs = [x for x in xlsxs if x.stem.lower() in want]
    if not xlsxs:
        raise SystemExit(f'No xlsx found under {args.metric_dir}')

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for xlsx in xlsxs:
        print(f'Processing {xlsx.name} ...')
        all_rows.extend(process_workbook(xlsx, args.out))

    write_summary(args.out, all_rows)
    print(f'Done. Wrote {args.out}/')
    print(f'  per-dataset: method_means.csv, ranges.csv, scores_A..F.csv, rankings.md')
    print(f'  summary: all_scores.csv, rank_matrix_{{A..F}}.csv, llvip_ranges_check.md')


if __name__ == '__main__':
    main()

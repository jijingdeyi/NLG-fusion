#!/usr/bin/env python3
"""Quality-control report + reannotation / conflict queues (score.md §3.3, §5.2).

  python scripts/pref_bench/qc_report.py --config /data/ykx/pref_bench/config.yaml
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pref.active import build_annotation_queue, low_reliability_pairs
from pref.io import export_supervision_jsonl, load_jsonl, load_pairs, load_yaml
from pref.qc import detect_cycles, prefer_edge, quality_report, sample_reannotation_pairs
from pref.schema import Label


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--reann-frac', type=float, default=0.15)
    ap.add_argument('--r-thresh', type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out = Path(cfg['out_root'])
    eta = float(cfg.get('eta', 0.5))
    superv_path = export_supervision_jsonl(out, eta=eta, cfg=cfg)
    rows = load_jsonl(superv_path)
    pairs = load_pairs(out)

    # flatten labels for tie ratio
    all_labs: list[Label] = []
    for r in rows:
        for _u, cnt in (r.get('per_annotator_counts') or {}).items():
            all_labs.extend(['A'] * cnt['A'] + ['Tie'] * cnt['Tie'] + ['B'] * cnt['B'])

    # cycles per scene from y
    by_scene = defaultdict(dict)
    for r in rows:
        edge = prefer_edge((r['y_A'], r['y_T'], r['y_B']))
        if not edge:
            continue
        a, b = r.get('candidate_a'), r.get('candidate_b')
        if not a or not b:
            p = pairs.get(r['pair_id'])
            if not p:
                continue
            a, b = p['candidate_a'], p['candidate_b']
        by_scene[r.get('scene_id', '')][(a, b)] = edge

    cycles = []
    for sid, edges in by_scene.items():
        for cyc in detect_cycles(edges):
            cycles.append({'scene_id': sid, 'cycle': cyc})

    report = quality_report(
        all_labs,
        cycle_count=len(cycles),
        n_pairs=max(len(rows), 1),
    )
    report['n_pairs'] = len(rows)
    report['n_cycles'] = len(cycles)
    report['mean_R'] = sum(r['R'] for r in rows) / len(rows) if rows else float('nan')

    reann = sample_reannotation_pairs([r['pair_id'] for r in rows], frac=args.reann_frac)
    low_r = low_reliability_pairs(rows, r_thresh=args.r_thresh)
    queue = build_annotation_queue(low_r, uncertain=[], coverage=[], reann_budget=len(low_r))

    fit = out / 'fit'
    fit.mkdir(parents=True, exist_ok=True)
    (fit / 'qc_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (fit / 'reannotation_queue.json').write_text(
        json.dumps({'random_reann': reann, **queue, 'cycles': cycles}, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(report, indent=2))
    print(f'wrote {fit / "qc_report.json"}')
    print(f'wrote {fit / "reannotation_queue.json"}')


if __name__ == '__main__':
    main()

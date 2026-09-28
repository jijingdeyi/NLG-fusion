#!/usr/bin/env python3
"""External human-alignment eval skeleton (score.md §5.4).

Maps a frozen scorer to public preference DBs (VISTA / LPIFM / EVAFusion).
Fill `adapters` once each external DB is locally available.

  python scripts/pref_bench/eval_external.py \\
      --ckpt /data/ykx/pref_bench/fit/scorer_best.pt \\
      --list
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


ADAPTERS = {
    'VISTA': {
        'status': 'TODO',
        'note': '200 scenes / 4000 fusions; need FQSV or pairwise export adapter',
    },
    'LPIFM': {
        'status': 'TODO',
        'note': 'independent A/B/Tie preference; convert to (IR,VI,F_A,F_B,y)',
    },
    'EVAFusion': {
        'status': 'TODO',
        'note': '850 source pairs / 9350 fusions; expert seed + LLM-expanded labels',
    },
}


def pairwise_acc_from_scores(rows: list[dict]) -> float:
    """rows: each has s_a, s_b, label in {A,B} (ties skipped)."""
    n = hit = 0
    for r in rows:
        if r['label'] == 'Tie':
            continue
        n += 1
        pred = 'A' if r['s_a'] > r['s_b'] else 'B'
        hit += int(pred == r['label'])
    return hit / n if n else float('nan')


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument('--ckpt', type=Path, default=None)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--dataset', choices=list(ADAPTERS), default=None)
    args = ap.parse_args()

    if args.list or args.dataset is None:
        print(json.dumps(ADAPTERS, indent=2, ensure_ascii=False))
        print('\nAdapters are stubs until external DB roots are configured.')
        return

    info = ADAPTERS[args.dataset]
    if info['status'] != 'ready':
        raise SystemExit(
            f'{args.dataset} adapter not ready: {info["note"]}\n'
            'Implement a loader that yields scored pairwise rows, then call '
            'pairwise_acc_from_scores(...).'
        )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Export pair-level supervision (y, R) from labels_*.jsonl (score.md §3.1.4–3.1.5).

  python scripts/pref_bench/aggregate_preferences.py \\
      --config /data/ykx/pref_bench/config.yaml
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pref.io import export_supervision_jsonl, load_yaml


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None, help='default: {out_root}/supervision.jsonl')
    ap.add_argument('--eta', type=float, default=None, help='non-professional weight η∈(0,1)')
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_root = Path(cfg['out_root'])
    eta = float(args.eta if args.eta is not None else cfg.get('eta', 0.5))
    dest = export_supervision_jsonl(out_root, dest=args.out, eta=eta, cfg=cfg)
    print(f'wrote {dest}')


if __name__ == '__main__':
    main()

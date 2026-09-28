#!/usr/bin/env python3
"""Score a single fusion or all candidates in a scene (score.md §4.5).

  python scripts/pref_bench/score_fusion.py \\
      --ckpt /data/ykx/pref_bench/fit/scorer_best.pt \\
      --ir IR.png --vi VI.png --fusion F.png
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pref.infer import load_scorer, score_fusion, score_scene_candidates


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--ir', type=Path, required=True)
    ap.add_argument('--vi', type=Path, required=True)
    ap.add_argument('--fusion', type=Path, default=None, help='single fusion path')
    ap.add_argument('--candidates-json', type=Path, default=None,
                    help='JSON object {name: path} for O(K) scene scoring')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    model, ckpt = load_scorer(args.ckpt, device=args.device)
    size = int(ckpt.get('size', 128))
    if args.candidates_json:
        cands = json.loads(args.candidates_json.read_text(encoding='utf-8'))
        scores = score_scene_candidates(
            model, str(args.ir), str(args.vi), cands, size=size, device=args.device,
        )
        print(json.dumps(scores, indent=2))
        best = max(scores, key=scores.get)
        print(f'top1={best} S={scores[best]:.4f}')
    else:
        if args.fusion is None:
            raise SystemExit('provide --fusion or --candidates-json')
        s = score_fusion(model, str(args.ir), str(args.vi), str(args.fusion),
                         size=size, device=args.device)
        print(f'S={s:.6f}')


if __name__ == '__main__':
    main()

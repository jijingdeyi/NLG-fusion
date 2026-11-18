#!/usr/bin/env python3
"""Score saved checkpoints on the in-domain test split and RoadScene.

Selection stays on the validation loss. This script only reports the two
held-out sets. RoadScene is also split into seen methods (e1) and unseen
methods (e2).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from pref.dataset import BaselinePairDataset, baseline_collate, make_scene_loader
from pref.io import load_jsonl
from pref.model.baseline import BaselineScorer
from pref.model.feature_scorer import FeatureScorer
from pref.model.region_scorer import RegionScorer
from pref.train_stage1 import eval_rank

_MODELS = {
    'baseline_ycbcr_tiles': BaselineScorer,
    'region_ycbcr': RegionScorer,
    'feature_ycbcr': FeatureScorer,
}


def _load(path: Path, device: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    kind = _MODELS[ckpt['arch']]
    model = kind(pretrained=False)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return model


def _rows(path: Path) -> list[dict]:
    return [row for row in load_jsonl(path) if row.get('ir') and row.get('path_a') and row.get('path_b')]


def _metrics(model, rows, device: str) -> dict:
    if not rows:
        return {'n': 0, 'loss': float('nan'), 'acc_soft': float('nan'), 'acc_decisive': float('nan'), 'n_decisive': 0}
    ds = BaselinePairDataset(rows, augment=False, exclude_gold=False, exclude_control=False)
    loader = make_scene_loader(ds, 8, shuffle=False, num_workers=2, collate_fn=baseline_collate)
    return eval_rank(model, loader, device)


def main() -> None:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    indomain = [row for row in _rows(Path('/data/ykx/pref_bench/supervision.jsonl')) if row.get('split') == 'test']
    road = _rows(Path('/data/ykx/pref_bench_roadscene/supervision.jsonl'))
    sets = {
        'indomain_test': indomain,
        'roadscene': road,
        'roadscene_e1': [row for row in road if str(row.get('pair_id', '')).startswith('e1::')],
        'roadscene_e2': [row for row in road if str(row.get('pair_id', '')).startswith('e2::')],
    }
    checkpoints = [
        Path('/data/ykx/pref_bench/fit/scorer_baseline_best.pt'),
        Path('/data/ykx/pref_bench/fit/scorer_region_best.pt'),
        Path('/data/ykx/pref_bench/fit/scorer_feature_best.pt'),
    ]
    for path in checkpoints:
        if not path.is_file():
            print(f'missing {path}', flush=True)
            continue
        model = _load(path, device)
        print(f'[{path.name}]', flush=True)
        for name, rows in sets.items():
            stats = _metrics(model, rows, device)
            print(
                f'  {name} n={stats["n"]} decisive={stats["n_decisive"]} '
                f'loss={stats["loss"]:.4f} acc_dec={stats["acc_decisive"]:.4f} '
                f'acc_soft={stats["acc_soft"]:.4f}',
                flush=True,
            )


if __name__ == '__main__':
    main()

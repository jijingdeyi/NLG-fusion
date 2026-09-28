#!/usr/bin/env python3
"""Cache classical fusion metrics for every (scene, candidate).

  python scripts/pref_bench/compute_features.py --config /data/ykx/pref_bench/config.yaml

Writes append/resume JSONL: {out_root}/features.jsonl
Keys: mi, qabf, scd, vif, ssim, sf
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ldt.fusion_metrics import ALL_METRIC_KEYS, compute_fusion_metrics


def load_config(path: Path) -> dict:
    with path.open(encoding='utf-8') as f:
        return yaml.safe_load(f)


def _read_gray(path: Path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return im.astype('float32')


def _done_keys(path: Path) -> set[tuple[str, str]]:
    done = set()
    if not path.is_file():
        return done
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row['scene_id'], row['candidate']))
    return done


def _job(payload: dict) -> dict:
    ir = _read_gray(Path(payload['ir']))
    vi = _read_gray(Path(payload['vi']))
    fused = _read_gray(Path(payload['fusion']))
    if fused.shape != ir.shape:
        fused = cv2.resize(fused, (ir.shape[1], ir.shape[0]), interpolation=cv2.INTER_AREA)
    m = compute_fusion_metrics(ir, vi, fused)
    row = {
        'scene_id': payload['scene_id'],
        'candidate': payload['candidate'],
        'dataset': payload['dataset'],
        'split': payload['split'],
        'fusion_path': payload['fusion'],
    }
    for k in ALL_METRIC_KEYS:
        row[k] = float(m[k])
    return row


def main() -> None:
    p = ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg['out_root'])
    scenes = []
    with (out / 'scenes.jsonl').open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                scenes.append(json.loads(line))
    cand_names = [c['name'] for c in cfg['candidates']]
    feat_path = out / 'features.jsonl'
    done = _done_keys(feat_path)

    jobs = []
    for sc in scenes:
        scene_dir = out / 'scenes' / sc['scene_id']
        ir = scene_dir / 'ir' / sc['filename']
        vi = scene_dir / 'vi' / sc['filename']
        for name in cand_names:
            if (sc['scene_id'], name) in done:
                continue
            fus = scene_dir / 'fusions' / f'{name}.png'
            if not fus.is_file():
                print(f'[skip] missing {fus}')
                continue
            jobs.append({
                'scene_id': sc['scene_id'],
                'candidate': name,
                'dataset': sc['dataset'],
                'split': sc['split'],
                'ir': str(ir),
                'vi': str(vi),
                'fusion': str(fus),
            })

    print(f'[features] todo={len(jobs)} done={len(done)} → {feat_path}')
    if not jobs:
        return

    n_ok = 0
    with feat_path.open('a', encoding='utf-8') as fout:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [ex.submit(_job, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                fout.write(json.dumps(row, ensure_ascii=False) + '\n')
                fout.flush()
                n_ok += 1
                if i % 20 == 0 or i == len(futs):
                    print(f'[features] {i}/{len(futs)}')
    print(f'[features] wrote +{n_ok} → {feat_path}')


if __name__ == '__main__':
    main()

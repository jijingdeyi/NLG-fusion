#!/usr/bin/env python3
"""Group near-duplicate scenes and keep each group in one split (score.md §3.3.3).

  PYTHONPATH=. python scripts/pref_bench/assign_scene_groups.py \\
      --config /data/ykx/pref_bench/config.yaml

Existing folds are kept. A group that currently spans train/val/test is moved
entirely into its majority split.
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pref.splits import DEFAULT_NEAR_DUP_MAE, apply_group_split


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def propagate(path: Path, splits: dict[str, str], groups: dict[str, str]) -> int:
    rows = load_jsonl(path)
    changed = 0
    for row in rows:
        sid = row.get('scene_id')
        if sid not in splits:
            continue
        new_split = splits[sid]
        if row.get('split') != new_split:
            changed += 1
        row['split'] = new_split
        if sid in groups:
            row['scene_group'] = groups[sid]
    if rows:
        write_jsonl(path, rows)
    return changed


def main() -> None:
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=Path('/data/ykx/pref_bench/config.yaml'))
    ap.add_argument('--threshold', type=float, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    out = Path(cfg['out_root'])
    threshold = float(args.threshold if args.threshold is not None else cfg.get('near_dup_mae', DEFAULT_NEAR_DUP_MAE))
    scenes = load_jsonl(out / 'scenes.jsonl')
    before = {s['scene_id']: s.get('split') for s in scenes}
    apply_group_split(scenes, threshold=threshold, repair=True)
    groups = {s['scene_id']: s['scene_group'] for s in scenes}
    splits = {s['scene_id']: s['split'] for s in scenes}
    write_jsonl(out / 'scenes.jsonl', scenes)

    moved = [sid for sid, old in before.items() if splits[sid] != old]
    members = defaultdict(list)
    for s in scenes:
        members[s['scene_group']].append(s['source_scene_id'] if 'source_scene_id' in s else s['scene_id'])
    multi = {gid: ids for gid, ids in members.items() if len(ids) > 1}

    for s in scenes:
        meta = out / 'scenes' / s['scene_id'] / 'meta.json'
        if not meta.is_file():
            continue
        payload = json.loads(meta.read_text(encoding='utf-8'))
        payload['split'] = s['split']
        payload['scene_group'] = s['scene_group']
        meta.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    touched = {'scenes.jsonl': len(moved)}
    for name in ('pairs.jsonl', 'annotations.jsonl', 'supervision.jsonl'):
        touched[name] = propagate(out / name, splits, groups)
    for fp in sorted(out.glob('labels_*.jsonl')):
        touched[fp.name] = propagate(fp, splits, groups)

    print(f'threshold MAE<={threshold}')
    print(f'near-duplicate groups: {len(multi)}')
    for gid, ids in sorted(multi.items(), key=lambda kv: kv[0]):
        print(f'  {gid}: {ids} -> {splits[gid.split("::", 1)[1]]}')
    print(f'scenes moved: {len(moved)}')
    for sid in moved:
        print(f'  {sid}: {before[sid]} -> {splits[sid]}')
    print('split counts', dict(Counter(splits.values())))
    print('rows whose split field changed', touched)


if __name__ == '__main__':
    main()

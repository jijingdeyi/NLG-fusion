#!/usr/bin/env python3
"""Migrate legacy pref_bench JSONL → paper schema (score.md §3.1.3–3.1.5).

Creates/updates under {out_root}:
  legacy_v1/          backup of pre-migration files
  scenes.jsonl        Scene(+source_*)
  candidates.jsonl    Candidate
  pairs.jsonl         Pair with fixed A/B ids
  annotators.jsonl    Annotator
  annotations.jsonl   canonical Annotation store
  labels_ann0.jsonl   enriched (UI-compatible) labels
  supervision.jsonl   derived (y, R) cache
  MANIFEST_v2.json    migration summary

Usage:
  PYTHONPATH=. python scripts/pref_bench/migrate_to_v2.py \\
      --config /data/ykx/pref_bench/config.yaml
"""

from __future__ import annotations

import json
import shutil
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pref.io import export_supervision_jsonl
from pref.schema import label_from_legacy_row, pair_fixed_ab, response_to_label


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def cand_id(scene_id: str, method_id: str) -> str:
    return f'{scene_id}::{method_id}'


def method_path(out: Path, scene_id: str, method_id: str) -> str:
    return str(out / 'scenes' / scene_id / 'fusions' / f'{method_id}.png')


def ir_vi_paths(out: Path, scene: dict) -> tuple[str, str]:
    sid = scene['scene_id']
    fn = scene.get('filename') or f"{sid.split('/')[-1]}.png"
    stem = Path(fn).name
    return (
        str(out / 'scenes' / sid / 'ir' / stem),
        str(out / 'scenes' / sid / 'vi' / stem),
    )


def migrate(out: Path, cfg: dict, eta: float) -> dict:
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = out / 'legacy_v1' / ts
    backup.mkdir(parents=True, exist_ok=True)

    to_backup = [
        'scenes.jsonl', 'pairs.jsonl', 'labels_ann0.jsonl', 'config.yaml',
    ]
    for name in to_backup:
        src = out / name
        if src.is_file():
            shutil.copy2(src, backup / name)

    scenes_raw = load_jsonl(out / 'scenes.jsonl')
    pairs_raw = load_jsonl(out / 'pairs.jsonl')
    label_files = sorted(out.glob('labels_*.jsonl'))
    labels_raw = []
    for fp in label_files:
        for row in load_jsonl(fp):
            row = dict(row)
            row.setdefault('annotator', fp.stem[len('labels_'):])
            labels_raw.append(row)

    scene_by_id = {s['scene_id']: s for s in scenes_raw}

    # ---- Scenes ----
    scenes_out = []
    for s in scenes_raw:
        sid = s['scene_id']
        ds = s.get('dataset') or sid.split('/')[0]
        fn = s.get('filename') or f"{sid.split('/')[-1]}.png"
        source_scene_id = Path(fn).stem
        ir, vi = ir_vi_paths(out, s)
        meta_path = out / 'scenes' / sid / 'meta.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.is_file() else {}
        scenes_out.append({
            'scene_id': sid,
            'source_dataset': ds,
            'source_scene_id': source_scene_id,
            'dataset': ds,  # backward compat
            'filename': fn,
            'root': s.get('root'),
            'split': s.get('split'),
            'ir': s.get('ir') or meta.get('ir') or ir,
            'vi': s.get('vi') or meta.get('vi') or vi,
            'scene_dir': str(out / 'scenes' / sid),
        })

    # ---- Candidates ----
    candidates_out = []
    cand_index: dict[str, dict] = {}
    for s in scenes_out:
        sid = s['scene_id']
        meta_path = out / 'scenes' / sid / 'meta.json'
        methods = []
        if meta_path.is_file():
            methods = json.loads(meta_path.read_text(encoding='utf-8')).get('candidates') or []
        if not methods:
            methods = [
                'prior_max', 'prior_mhf', 'method_a', 'method_b',
                'method_c', 'method_d', 'deg_overexp', 'deg_underexp',
            ]
        for mid in methods:
            cid = cand_id(sid, mid)
            row = {
                'candidate_id': cid,
                'scene_id': sid,
                'method_id': mid,
                'path': method_path(out, sid, mid),
            }
            candidates_out.append(row)
            cand_index[cid] = row

    # ---- Pairs (fixed A/B) ----
    pair_by_id: dict[str, dict] = {}
    pairs_out = []
    for p in pairs_raw:
        pid = p.get('pair_id') or p['task_id']
        scene_id = p['scene_id']
        left, right = p['left'], p['right']
        a_method, b_method = pair_fixed_ab({'left': left, 'right': right})
        a_id, b_id = cand_id(scene_id, a_method), cand_id(scene_id, b_method)
        scene = scene_by_id.get(scene_id, {})
        ir = p.get('ir') or ir_vi_paths(out, {'scene_id': scene_id, 'filename': scene.get('filename')})[0]
        vi = p.get('vi') or ir_vi_paths(out, {'scene_id': scene_id, 'filename': scene.get('filename')})[1]
        # paths for A/B (fixed) and left/right (display)
        path_a = method_path(out, scene_id, a_method)
        path_b = method_path(out, scene_id, b_method)
        row = {
            'pair_id': pid,
            'task_id': p['task_id'],  # legacy alias
            'scene_id': scene_id,
            'dataset': p.get('dataset') or scene.get('dataset'),
            'split': p.get('split') or scene.get('split'),
            'candidate_a_id': a_id,
            'candidate_b_id': b_id,
            'candidate_a': a_method,
            'candidate_b': b_method,
            'left': left,
            'right': right,
            'left_path': p.get('left_path') or method_path(out, scene_id, left),
            'right_path': p.get('right_path') or method_path(out, scene_id, right),
            'path_a': path_a,
            'path_b': path_b,
            'ir': ir,
            'vi': vi,
            'is_gold': bool(p.get('is_gold', False)),
            'gold_expect': p.get('gold_expect'),
        }
        pairs_out.append(row)
        pair_by_id[pid] = row

    # Reconstruct orphan labeled pairs missing from pairs.jsonl
    orphan_added = 0
    for lab in labels_raw:
        tid = lab['task_id']
        if tid in pair_by_id:
            continue
        scene_id = lab['scene_id']
        left, right = lab['left'], lab['right']
        scene = scene_by_id.get(scene_id)
        if scene is None:
            # synthesize minimal scene from label
            ds = lab.get('dataset') or scene_id.split('/')[0]
            scene = {
                'scene_id': scene_id,
                'dataset': ds,
                'filename': f"{scene_id.split('/')[-1]}.png",
                'split': lab.get('split', 'train'),
            }
            scene_by_id[scene_id] = scene
            if not any(s['scene_id'] == scene_id for s in scenes_out):
                ir, vi = ir_vi_paths(out, scene)
                scenes_out.append({
                    'scene_id': scene_id,
                    'source_dataset': ds,
                    'source_scene_id': Path(scene['filename']).stem,
                    'dataset': ds,
                    'filename': scene['filename'],
                    'root': None,
                    'split': scene['split'],
                    'ir': ir,
                    'vi': vi,
                    'scene_dir': str(out / 'scenes' / scene_id),
                })
        a_method, b_method = pair_fixed_ab({'left': left, 'right': right})
        a_id, b_id = cand_id(scene_id, a_method), cand_id(scene_id, b_method)
        ir, vi = ir_vi_paths(out, scene)
        row = {
            'pair_id': tid,
            'task_id': tid,
            'scene_id': scene_id,
            'dataset': lab.get('dataset') or scene.get('dataset'),
            'split': lab.get('split') or scene.get('split'),
            'candidate_a_id': a_id,
            'candidate_b_id': b_id,
            'candidate_a': a_method,
            'candidate_b': b_method,
            'left': left,
            'right': right,
            'left_path': method_path(out, scene_id, left),
            'right_path': method_path(out, scene_id, right),
            'path_a': method_path(out, scene_id, a_method),
            'path_b': method_path(out, scene_id, b_method),
            'ir': ir,
            'vi': vi,
            'is_gold': bool(lab.get('is_gold', False)),
            'gold_expect': lab.get('gold_expect'),
            'recovered_from_label': True,
        }
        pairs_out.append(row)
        pair_by_id[tid] = row
        orphan_added += 1
        # ensure candidates exist
        for mid in (left, right, a_method, b_method):
            cid = cand_id(scene_id, mid)
            if cid not in cand_index:
                crow = {
                    'candidate_id': cid,
                    'scene_id': scene_id,
                    'method_id': mid,
                    'path': method_path(out, scene_id, mid),
                }
                candidates_out.append(crow)
                cand_index[cid] = crow

    # ---- Annotators ----
    ann_ids = sorted({lab.get('annotator') or lab.get('annotator_id') or 'ann0' for lab in labels_raw})
    if not ann_ids:
        ann_ids = list(cfg.get('annotator_ids') or ['ann0'])
    # Fill expertise: sole historical annotator treated as professional (IVIF expert labeling)
    cfg_ann = {
        (a['annotator_id'] if isinstance(a, dict) else str(a)): (
            a.get('expertise', 'professional') if isinstance(a, dict) else 'professional'
        )
        for a in (cfg.get('annotators') or [])
    }
    for aid in cfg.get('annotator_ids') or []:
        cfg_ann.setdefault(str(aid), 'professional')
    annotators_out = []
    for aid in ann_ids:
        annotators_out.append({
            'annotator_id': aid,
            'expertise': cfg_ann.get(aid, 'professional'),
            'note': 'migrated from legacy labels; default expertise=professional',
        })
    # also register unused configured annotators for future rounds
    for aid, exp in cfg_ann.items():
        if aid not in ann_ids:
            annotators_out.append({
                'annotator_id': aid,
                'expertise': exp,
                'note': 'configured but no labels yet',
            })

    # ---- Annotations ----
    annotations_out = []
    labels_enriched = {aid: [] for aid in ann_ids}
    round_counter: dict[tuple[str, str], int] = {}

    for lab in labels_raw:
        tid = lab['task_id']
        pair = pair_by_id.get(tid)
        if pair is None:
            continue
        aid = lab.get('annotator') or lab.get('annotator_id') or 'ann0'
        key = (aid, tid)
        round_counter[key] = round_counter.get(key, 0) + 1
        round_id = int(lab.get('round_id') or round_counter[key])
        left = lab.get('left_candidate') or lab.get('left') or pair['left']
        right = lab.get('right_candidate') or lab.get('right') or pair['right']
        response = lab.get('response') or lab.get('choice')
        if response not in ('left', 'right', 'tie'):
            raise ValueError(f'bad response for {tid}: {response}')
        label = lab.get('label')
        if label not in ('A', 'Tie', 'B'):
            label = response_to_label(
                response,
                left, right,
                pair['candidate_a'], pair['candidate_b'],
            )
        ann_id = lab.get('annotation_id') or f'{aid}:{tid}:r{round_id}'
        ann = {
            'annotation_id': ann_id,
            'pair_id': pair['pair_id'],
            'task_id': tid,
            'annotator_id': aid,
            'round_id': round_id,
            'left_candidate': left,
            'right_candidate': right,
            'response': response,
            'label': label,
            'ts': lab.get('ts'),
            'scene_id': pair['scene_id'],
            'dataset': pair.get('dataset'),
            'split': pair.get('split'),
            'is_gold': bool(pair.get('is_gold', False)),
        }
        if lab.get('gold_expect') is not None:
            ann['gold_expect'] = lab['gold_expect']
            ann['gold_ok'] = lab.get('gold_ok')
        annotations_out.append(ann)

        # UI-compatible enriched label row
        enr = {
            'task_id': tid,
            'pair_id': pair['pair_id'],
            'annotation_id': ann_id,
            'annotator': aid,
            'annotator_id': aid,
            'round_id': round_id,
            'choice': response,
            'response': response,
            'label': label,
            'left': left,
            'right': right,
            'left_candidate': left,
            'right_candidate': right,
            'candidate_a': pair['candidate_a'],
            'candidate_b': pair['candidate_b'],
            'ts': lab.get('ts'),
            'scene_id': pair['scene_id'],
            'dataset': pair.get('dataset'),
            'split': pair.get('split'),
            'is_gold': bool(pair.get('is_gold', False)),
        }
        if lab.get('gold_expect') is not None:
            enr['gold_expect'] = lab['gold_expect']
            enr['gold_ok'] = lab.get('gold_ok')
        labels_enriched.setdefault(aid, []).append(enr)

    # ---- Write ----
    write_jsonl(out / 'scenes.jsonl', scenes_out)
    write_jsonl(out / 'candidates.jsonl', candidates_out)
    write_jsonl(out / 'pairs.jsonl', pairs_out)
    write_jsonl(out / 'annotators.jsonl', annotators_out)
    write_jsonl(out / 'annotations.jsonl', annotations_out)
    for aid, rows in labels_enriched.items():
        write_jsonl(out / f'labels_{aid}.jsonl', rows)

    # update config with annotators + eta if missing
    cfg_path = out / 'config.yaml'
    cfg2 = dict(cfg)
    cfg2['out_root'] = str(out)
    cfg2['eta'] = float(cfg.get('eta', eta))
    if not cfg2.get('annotators'):
        cfg2['annotators'] = [
            {'annotator_id': a['annotator_id'], 'expertise': a['expertise']}
            for a in annotators_out
        ]
    if not cfg2.get('annotator_ids'):
        cfg2['annotator_ids'] = [a['annotator_id'] for a in annotators_out]
    cfg_path.write_text(
        yaml.safe_dump(cfg2, sort_keys=False, allow_unicode=True),
        encoding='utf-8',
    )

    # derived supervision
    superv = export_supervision_jsonl(out, dest=out / 'supervision.jsonl', eta=cfg2['eta'], cfg=cfg2)

    manifest = {
        'migrated_at': datetime.now(timezone.utc).isoformat(),
        'backup': str(backup),
        'n_scenes': len(scenes_out),
        'n_candidates': len(candidates_out),
        'n_pairs': len(pairs_out),
        'n_pairs_recovered': orphan_added,
        'n_annotators': len(annotators_out),
        'n_annotations': len(annotations_out),
        'supervision': str(superv),
        'eta': cfg2['eta'],
        'schema': 'score.md §3.1.3 v2',
        'notes': [
            'candidate_a/b fixed by lexicographic method_id order',
            'ann0 expertise defaulted to professional',
            'orphan labels recovered into pairs.jsonl',
            'supervision.jsonl is derived cache of (y,R); re-run aggregate after new labels',
        ],
    }
    (out / 'MANIFEST_v2.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


def main() -> None:
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=Path('/data/ykx/pref_bench/config.yaml'))
    ap.add_argument('--out-root', type=Path, default=None)
    ap.add_argument('--eta', type=float, default=0.5)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    out = Path(args.out_root or cfg.get('out_root') or args.config.parent).resolve()
    manifest = migrate(out, cfg, eta=args.eta)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

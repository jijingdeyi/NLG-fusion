"""I/O helpers bridging existing pref_bench JSONL ↔ (y, R) supervision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from pref.aggregate import PairSupervision, aggregate_all_pairs
from pref.schema import Expertise, Label, label_from_legacy_row, pair_fixed_ab


def load_yaml(path: Path | str) -> dict:
    with Path(path).open(encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_PATH_KEYS = (
    'ir', 'vi', 'path_a', 'path_b', 'left_path', 'right_path', 'fusion', 'path',
)


def resolve_pack_paths(rows: list[dict], out_root: Path | str) -> list[dict]:
    """Join relative media paths against the preference-pack root (portable releases)."""
    root = Path(out_root).resolve()
    for r in rows:
        for k in _PATH_KEYS:
            v = r.get(k)
            if not v or not isinstance(v, str):
                continue
            p = Path(v)
            if not p.is_absolute():
                r[k] = str((root / p).resolve())
    return rows


def append_jsonl(path: Path | str, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_pairs(out_root: Path | str) -> dict[str, dict]:
    """task_id / pair_id → pair row."""
    out = Path(out_root)
    pairs = {}
    for row in load_jsonl(out / 'pairs.jsonl'):
        pid = row.get('pair_id') or row['task_id']
        row = dict(row)
        row['pair_id'] = pid
        a, b = pair_fixed_ab(row)
        row['candidate_a'] = a
        row['candidate_b'] = b
        pairs[pid] = row
    return pairs


def load_annotators(out_root: Path | str, cfg: Mapping | None = None) -> dict[str, Expertise]:
    """Map annotator_id → expertise.

    Priority: {out_root}/annotators.jsonl > config['annotators'] > default professional.
    """
    out = Path(out_root)
    expertise: dict[str, Expertise] = {}
    for row in load_jsonl(out / 'annotators.jsonl'):
        expertise[row['annotator_id']] = row.get('expertise', 'professional')
    if cfg:
        for row in cfg.get('annotators', []) or []:
            if isinstance(row, dict):
                expertise.setdefault(row['annotator_id'], row.get('expertise', 'professional'))
            else:
                expertise.setdefault(str(row), 'professional')
        for aid in cfg.get('annotator_ids', []) or []:
            expertise.setdefault(str(aid), 'professional')
    return expertise


def iter_legacy_annotations(
    out_root: Path | str,
    pairs: Mapping[str, dict] | None = None,
    annotator: str | None = None,
) -> list[tuple[str, str, Label, dict]]:
    """Load annotations as (pair_id, annotator_id, label, raw_row).

    Prefer canonical `annotations.jsonl` if present; else fall back to labels_*.jsonl.
    Keeps every annotation (multi-round).
    """
    out = Path(out_root)
    pairs = pairs or load_pairs(out)
    canon = out / 'annotations.jsonl'
    rows: list[tuple[str, str, Label, dict]] = []
    if canon.is_file():
        for row in load_jsonl(canon):
            aid = row.get('annotator_id') or row.get('annotator')
            if annotator is not None and aid != annotator:
                continue
            pid = row.get('pair_id') or row.get('task_id')
            lab = row.get('label')
            if lab not in ('A', 'Tie', 'B'):
                pair = pairs.get(pid)
                if pair is None:
                    continue
                lab = label_from_legacy_row(pair, {
                    'choice': row.get('response') or row.get('choice'),
                    'left': row.get('left_candidate') or row.get('left'),
                    'right': row.get('right_candidate') or row.get('right'),
                })
            rows.append((pid, aid, lab, row))  # type: ignore[arg-type]
        return rows

    files = sorted(out.glob('labels_*.jsonl')) if annotator is None else [out / f'labels_{annotator}.jsonl']
    for fp in files:
        if not fp.is_file():
            continue
        ann_id = fp.stem[len('labels_'):]
        for row in load_jsonl(fp):
            tid = row['task_id']
            pair = pairs.get(tid) or pairs.get(row.get('pair_id', ''))
            if pair is None:
                continue
            lab = row.get('label')
            if lab not in ('A', 'Tie', 'B'):
                lab = label_from_legacy_row(pair, row)
            rows.append((pair.get('pair_id', tid), ann_id, lab, row))  # type: ignore[arg-type]
    return rows


def build_supervision(
    out_root: Path | str,
    eta: float = 0.5,
    cfg: Mapping | None = None,
    *,
    pool_annotations: bool = True,
) -> dict[str, PairSupervision]:
    out = Path(out_root)
    pairs = load_pairs(out)
    expertise = load_annotators(out, cfg)
    anns = [(pid, uid, lab) for pid, uid, lab, _ in iter_legacy_annotations(out, pairs)]
    return aggregate_all_pairs(
        anns, expertise=expertise, eta=eta, pool_annotations=pool_annotations,
    )


def supervision_to_row(sup: PairSupervision, pair: dict | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        'pair_id': sup.pair_id,
        'y_A': sup.y[0],
        'y_T': sup.y[1],
        'y_B': sup.y[2],
        'R': sup.R,
        'n_annotators': sup.n_annotators,
        'n_annotations': sup.n_annotations,
        'per_annotator_counts': {
            k: {'A': v[0], 'Tie': v[1], 'B': v[2]}
            for k, v in sup.per_annotator_counts.items()
        },
    }
    if pair:
        for k in ('scene_id', 'split', 'dataset', 'is_gold', 'candidate_a', 'candidate_b',
                  'left', 'right', 'ir', 'vi', 'left_path', 'right_path'):
            if k in pair:
                row[k] = pair[k]
        # paths aligned to fixed A/B
        if 'candidate_a' in pair and 'left' in pair:
            if pair['candidate_a'] == pair['left']:
                row['path_a'] = pair.get('left_path')
                row['path_b'] = pair.get('right_path')
            else:
                row['path_a'] = pair.get('right_path')
                row['path_b'] = pair.get('left_path')
    return row


def export_supervision_jsonl(
    out_root: Path | str,
    dest: Path | str | None = None,
    eta: float = 0.5,
    cfg: Mapping | None = None,
) -> Path:
    out = Path(out_root)
    dest = Path(dest) if dest else out / 'supervision.jsonl'
    pairs = load_pairs(out)
    supers = build_supervision(out, eta=eta, cfg=cfg)
    rows = [supervision_to_row(supers[pid], pairs.get(pid)) for pid in sorted(supers)]
    write_jsonl(dest, rows)
    return dest

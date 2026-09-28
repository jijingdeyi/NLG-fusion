"""Train/val/test splits with near-duplicate isolation (score.md §3.3.3)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

# Mean |IR,VI| difference on 64×64 thumbnails. Adjacent frames in this bench
# sit around 0.02–0.04; unrelated scenes sit around 0.15.
DEFAULT_NEAR_DUP_MAE = 0.05
_SPLIT_PRIORITY = ('train', 'val', 'test')


def image_pair_vector(ir_path: str | Path, vi_path: str | Path, size: int = 64) -> np.ndarray:
    """Concatenate downsampled grayscale IR and VI, values in [0, 1]."""

    def _load(path: str | Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32).ravel() / 255.0

    return np.concatenate([_load(ir_path), _load(vi_path)])


def resolve_source_paths(scene: Mapping) -> tuple[Path, Path] | None:
    ir, vi = scene.get('ir'), scene.get('vi')
    if ir and vi and Path(ir).is_file() and Path(vi).is_file():
        return Path(ir), Path(vi)
    root, filename = scene.get('root'), scene.get('filename')
    if root and filename:
        ir_p, vi_p = Path(root) / 'ir' / filename, Path(root) / 'vi' / filename
        if ir_p.is_file() and vi_p.is_file():
            return ir_p, vi_p
    return None


class _UnionFind:
    def __init__(self, items: Sequence[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_by_distance(
    scene_ids: Sequence[str],
    dataset_of: Mapping[str, str],
    vectors: Mapping[str, np.ndarray],
    threshold: float = DEFAULT_NEAR_DUP_MAE,
) -> dict[str, str]:
    """Link scenes in the same dataset whose mean absolute difference ≤ threshold.

    Group id is ``{dataset}::{lexicographically smallest scene_id in the component}``.
    """
    uf = _UnionFind(scene_ids)
    by_ds: dict[str, list[str]] = defaultdict(list)
    for sid in scene_ids:
        by_ds[dataset_of[sid]].append(sid)
    for members in by_ds.values():
        for i, a in enumerate(members):
            va = vectors[a]
            for b in members[i + 1:]:
                diff = float(np.mean(np.abs(va - vectors[b])))
                if diff <= threshold:
                    uf.union(a, b)
    groups: dict[str, list[str]] = defaultdict(list)
    for sid in scene_ids:
        groups[uf.find(sid)].append(sid)
    out = {}
    for members in groups.values():
        root = min(members)
        gid = f'{dataset_of[root]}::{root}'
        for sid in members:
            out[sid] = gid
    return out


def cluster_scenes(
    scenes: Sequence[dict],
    threshold: float = DEFAULT_NEAR_DUP_MAE,
) -> dict[str, str]:
    """scene_id → scene_group from source-image similarity."""
    ids = []
    dataset_of = {}
    vectors = {}
    for scene in scenes:
        sid = scene['scene_id']
        paths = resolve_source_paths(scene)
        if paths is None:
            continue
        ids.append(sid)
        dataset_of[sid] = scene.get('source_dataset') or scene.get('dataset') or sid.split('/')[0]
        vectors[sid] = image_pair_vector(*paths)
    grouped = cluster_by_distance(ids, dataset_of, vectors, threshold=threshold)
    for scene in scenes:
        sid = scene['scene_id']
        grouped.setdefault(sid, f"{dataset_of.get(sid, sid)}::{sid}")
    return grouped


def assign_splits_by_group(
    scene_ids: Sequence[str],
    group_of: Mapping[str, str] | None = None,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 443,
) -> dict[str, str]:
    """Split by scene-group so near-duplicates stay in the same fold.

    Groups are assigned inside each dataset (the prefix before '::').
    """
    group_of = group_of or {s: s for s in scene_ids}
    by_ds: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sid in scene_ids:
        gid = group_of[sid]
        ds = gid.split('::', 1)[0]
        by_ds[ds][gid].append(sid)
    out: dict[str, str] = {}
    rng = np.random.default_rng(seed)
    for ds, groups in by_ds.items():
        items = list(groups.items())
        rng.shuffle(items)
        items.sort(key=lambda kv: len(kv[1]), reverse=True)
        n = sum(len(v) for _, v in items)
        targets = [n * r for r in ratios]
        loads = [0, 0, 0]
        for gid, members in items:
            idx = max(range(3), key=lambda i: targets[i] - loads[i])
            loads[idx] += len(members)
            for sid in members:
                out[sid] = _SPLIT_PRIORITY[idx]
    return out


def repair_splits_by_group(
    scenes: Sequence[dict],
    group_of: Mapping[str, str],
) -> dict[str, str]:
    """Keep current folds, but pull a mixed group into one split.

    Majority split wins. Ties prefer train, then val, so a near-duplicate of a
    training scene does not stay in test.
    """
    members: dict[str, list[dict]] = defaultdict(list)
    for scene in scenes:
        members[group_of[scene['scene_id']]].append(scene)
    out = {}
    for gid, rows in members.items():
        counts = {name: 0 for name in _SPLIT_PRIORITY}
        for row in rows:
            counts[row.get('split') or 'train'] = counts.get(row.get('split') or 'train', 0) + 1
        best = sorted(_SPLIT_PRIORITY, key=lambda name: (-counts.get(name, 0), _SPLIT_PRIORITY.index(name)))[0]
        for row in rows:
            out[row['scene_id']] = best
    return out


def apply_group_split(
    scenes: list[dict],
    threshold: float = DEFAULT_NEAR_DUP_MAE,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 443,
    repair: bool = False,
) -> dict[str, str]:
    """Write ``scene_group`` and ``split`` onto scene dicts. Returns scene_id → split."""
    group_of = cluster_scenes(scenes, threshold=threshold)
    if repair:
        splits = repair_splits_by_group(scenes, group_of)
    else:
        splits = assign_splits_by_group([s['scene_id'] for s in scenes], group_of, ratios=ratios, seed=seed)
    for scene in scenes:
        sid = scene['scene_id']
        scene['scene_group'] = group_of[sid]
        scene['split'] = splits[sid]
    return splits


def mark_unseen_datasets(
    scenes: Sequence[dict],
    unseen_datasets: Sequence[str],
) -> dict[str, str]:
    """Freeze S* then evaluate on LLVIP / RoadScene / OpIVF etc."""
    unseen = set(unseen_datasets)
    out = {}
    for s in scenes:
        sid = s['scene_id']
        ds = s.get('dataset') or s.get('source_dataset')
        out[sid] = 'unseen' if ds in unseen else s.get('split', 'train')
    return out

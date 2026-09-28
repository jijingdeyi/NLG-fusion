"""Scene-level pair-graph connectivity checks (score.md §3.4 / Stage-1 prep)."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def _candidates(row: Mapping) -> tuple[str, str] | None:
    a = row.get('candidate_a') or row.get('left')
    b = row.get('candidate_b') or row.get('right')
    if not a or not b or a == b:
        return None
    return str(a), str(b)


def scene_pair_graph(rows: Sequence[Mapping]) -> dict[str, dict]:
    """Build undirected candidate graphs per scene from pair-level rows."""
    by_scene: dict[str, dict] = {}
    buckets: dict[str, list[Mapping]] = defaultdict(list)
    for r in rows:
        sid = str(r.get('scene_id') or '')
        if sid:
            buckets[sid].append(r)

    for sid, rs in buckets.items():
        nodes: set[str] = set()
        edges: list[tuple[str, str]] = []
        for r in rs:
            ab = _candidates(r)
            if ab is None:
                continue
            a, b = ab
            nodes.update((a, b))
            edges.append((a, b) if a <= b else (b, a))
        parent = {n: n for n in nodes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in edges:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        comps: dict[str, list[str]] = defaultdict(list)
        for n in nodes:
            comps[find(n)].append(n)
        by_scene[sid] = {
            'scene_id': sid,
            'n_pairs': len(rs),
            'n_edges': len(set(edges)),
            'n_candidates': len(nodes),
            'n_components': len(comps),
            'connected': len(comps) <= 1 and len(nodes) > 0,
            'components': [sorted(v) for v in comps.values()],
            'candidates': sorted(nodes),
        }
    return by_scene


def connectivity_report(
    rows: Sequence[Mapping],
    *,
    split: str | None = None,
) -> dict:
    """Summarize which scenes have disconnected / empty pair graphs."""
    use = [r for r in rows if split is None or r.get('split') == split]
    graphs = scene_pair_graph(use)
    bad = [g for g in graphs.values() if not g['connected']]
    return {
        'n_scenes': len(graphs),
        'n_pairs': len(use),
        'n_disconnected': len(bad),
        'disconnected': [
            {
                'scene_id': g['scene_id'],
                'n_candidates': g['n_candidates'],
                'n_edges': g['n_edges'],
                'n_components': g['n_components'],
                'components': g['components'],
            }
            for g in sorted(bad, key=lambda x: x['scene_id'])
        ],
    }

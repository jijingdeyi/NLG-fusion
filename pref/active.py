"""Incremental / active annotation helpers (score.md §3.4)."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


def low_reliability_pairs(
    supervision: Mapping[str, dict] | Sequence[dict],
    r_thresh: float = 0.5,
    topk: int | None = None,
) -> list[str]:
    """Re-annotation queue: existing pairs with low R (depth)."""
    rows = supervision.values() if isinstance(supervision, dict) else supervision
    scored = [(r['pair_id'], float(r['R'])) for r in rows]
    scored = [(pid, R) for pid, R in scored if R < r_thresh]
    scored.sort(key=lambda x: x[1])
    ids = [pid for pid, _ in scored]
    return ids[:topk] if topk is not None else ids


def uncertain_new_pairs(
    score_diff: Mapping[tuple[str, str], float],
    already: set[tuple[str, str]] | None = None,
    topk: int = 50,
) -> list[tuple[str, str]]:
    """New-pair annotation: smallest |s_i - s_j| (decision boundary)."""
    already = already or set()
    items = []
    for (a, b), d in score_diff.items():
        key = (a, b) if a <= b else (b, a)
        if key in already or (key[1], key[0]) in already:
            continue
        items.append((key[0], key[1], abs(float(d))))
    items.sort(key=lambda x: x[2])
    return [(a, b) for a, b, _ in items[:topk]]


def coverage_pairs(
    candidates: Sequence[str],
    already: set[tuple[str, str]] | None = None,
    per_cand: int = 2,
    seed: int = 443,
) -> list[tuple[str, str]]:
    """Ensure each candidate appears in some pairs (coverage)."""
    rng = np.random.default_rng(seed)
    already = already or set()
    cand = list(candidates)
    need = {c: per_cand for c in cand}
    for a, b in already:
        need[a] = max(0, need.get(a, 0) - 1)
        need[b] = max(0, need.get(b, 0) - 1)
    out = []
    for c in cand:
        while need.get(c, 0) > 0:
            others = [x for x in cand if x != c]
            if not others:
                break
            o = str(rng.choice(others))
            key = (c, o) if c <= o else (o, c)
            if key in already or key in out:
                need[c] -= 1
                continue
            out.append(key)
            need[c] -= 1
            need[o] = max(0, need.get(o, 0) - 1)
    return out


def build_annotation_queue(
    low_r: Sequence[str],
    uncertain: Sequence[tuple[str, str]],
    coverage: Sequence[tuple[str, str]],
    control: Sequence[tuple[str, str]] | None = None,
    new_pair_budget: int = 40,
    reann_budget: int = 20,
) -> dict:
    """Mix new-pair + re-annotation queues (§3.4)."""
    control = list(control or [])
    return {
        'reannotation_pair_ids': list(low_r)[:reann_budget],
        'new_pairs': {
            'uncertain': list(uncertain)[: new_pair_budget // 2],
            'coverage': list(coverage)[: max(1, new_pair_budget // 4)],
            'control': control[: max(1, new_pair_budget // 4)],
        },
    }


def all_candidate_pairs(candidates: Sequence[str]) -> list[tuple[str, str]]:
    cs = list(candidates)
    return [(cs[i], cs[j]) for i in range(len(cs)) for j in range(i + 1, len(cs))]

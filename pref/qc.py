"""Data cleaning & quality reports (score.md §3.3, §5.2)."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np

from pref.schema import LABEL_A, LABEL_B, LABEL_TIE, Label


def sample_reannotation_pairs(
    pair_ids: Sequence[str],
    frac: float = 0.15,
    seed: int = 443,
) -> list[str]:
    """§3.3.1 random re-annotation subset (~15–20%)."""
    rng = np.random.default_rng(seed)
    ids = list(pair_ids)
    n = max(1, int(round(len(ids) * frac))) if ids else 0
    if n >= len(ids):
        return ids
    idx = rng.choice(len(ids), size=n, replace=False)
    return [ids[i] for i in sorted(idx)]


def agreement_rate(labels_a: Sequence[Label], labels_b: Sequence[Label]) -> float:
    if not labels_a or len(labels_a) != len(labels_b):
        return float('nan')
    return float(np.mean([a == b for a, b in zip(labels_a, labels_b)]))


def cohens_kappa(labels_a: Sequence[Label], labels_b: Sequence[Label]) -> float:
    """Cohen's κ over {A,Tie,B}."""
    cats = [LABEL_A, LABEL_TIE, LABEL_B]
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        return float('nan')
    po = agreement_rate(labels_a, labels_b)
    ca, cb = Counter(labels_a), Counter(labels_b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0
    return float((po - pe) / (1.0 - pe))


def swap_consistency(
    labels_forward: Sequence[Label],
    labels_swapped: Sequence[Label],
) -> float:
    """After swapping display sides, A↔B should flip (Tie stays)."""
    if not labels_forward or len(labels_forward) != len(labels_swapped):
        return float('nan')
    ok = 0
    for a, b in zip(labels_forward, labels_swapped):
        if a == LABEL_TIE:
            ok += int(b == LABEL_TIE)
        elif a == LABEL_A:
            ok += int(b == LABEL_B)
        else:
            ok += int(b == LABEL_A)
    return ok / len(labels_forward)


def tie_ratio(labels: Sequence[Label]) -> float:
    if not labels:
        return float('nan')
    return float(np.mean([lab == LABEL_TIE for lab in labels]))


def prefer_edge(y: tuple[float, float, float], margin: float = 0.05) -> str | None:
    """Return 'A>B', 'B>A', or None (tie-ish)."""
    y_a, y_t, y_b = y
    if y_a >= y_b + margin and y_a >= y_t:
        return 'A>B'
    if y_b >= y_a + margin and y_b >= y_t:
        return 'B>A'
    return None


def detect_cycles(
    scene_edges: Mapping[str, str],
) -> list[tuple[str, str, str]]:
    """Find 3-cycles among candidates given directed preferences cand_i>cand_j.

    scene_edges: map 'A|B' (lex smaller first key unused) → 'A>B'/'B>A'
    Actually expect edges as dict[(u,v)] = True meaning u preferred over v.
    """
    # normalize to directed wins
    wins: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for (u, v), pref in scene_edges.items():
        nodes.update([u, v])
        if pref == 'A>B' or pref is True or pref == u:
            wins[u].add(v)
        elif pref == 'B>A' or pref == v:
            wins[v].add(u)
    cycles = []
    for a, b, c in combinations(sorted(nodes), 3):
        # check all 6 orientations? detect if cyclic tournament
        ab = b in wins[a]
        ba = a in wins[b]
        bc = c in wins[b]
        cb = b in wins[c]
        ca = a in wins[c]
        ac = c in wins[a]
        # cycle a>b>c>a
        if ab and bc and ca:
            cycles.append((a, b, c))
        if ac and cb and ba:
            cycles.append((a, c, b))
    return cycles


def conflict_pairs(
    supervision_rows: Sequence[dict],
    model_probs: Mapping[str, tuple[float, float, float]],
    conf_thresh: float = 0.9,
    human_margin: float = 0.2,
) -> list[str]:
    """§3.3.2 directed review: high-confidence model vs human disagreement."""
    out = []
    for row in supervision_rows:
        pid = row['pair_id']
        if pid not in model_probs:
            continue
        y = (row['y_A'], row['y_T'], row['y_B'])
        p = model_probs[pid]
        human = prefer_edge(y, margin=human_margin)
        # model mode
        mi = int(np.argmax(p))
        if p[mi] < conf_thresh:
            continue
        model_lab = ('A', 'Tie', 'B')[mi]
        if human == 'A>B' and model_lab == 'B':
            out.append(pid)
        elif human == 'B>A' and model_lab == 'A':
            out.append(pid)
    return out


def _pair_key(a: str, b: str) -> frozenset:
    return frozenset((a, b))


def recommend_review(
    supervision_rows: Sequence[dict],
    pairs: Sequence[dict],
    frac: float = 0.15,
    seed: int = 443,
    r_thresh: float = 0.5,
    max_queue: int = 80,
    low_r_budget: int = 40,
) -> dict:
    """Build a re-annotation queue from score.md §3.3–§3.4.

    Priority:
      1. pairs that sit on a detected preference cycle (§3.3.2) — edges only
      2. a small budget of lowest-R pairs (§3.4), only when R actually varies
      3. otherwise a random ~15% sample (§3.3.1), capped

    Cycle review is edge-local, not whole-scene: as K grows, C(K,2) would
    otherwise flood the queue with pairs that are not on any cycle.
    Do NOT dump every pair with R < r_thresh (first-round R≈0.25 for most).
    """
    # (scene, unordered method pair) → pair_id. Method names repeat across scenes.
    cand_pair_to_id: dict[tuple[str, frozenset], str] = {}
    scene_of: dict[str, str] = {}
    for p in pairs:
        pid = p.get('pair_id') or p['task_id']
        sid = p.get('scene_id', '')
        scene_of[pid] = sid
        scene_of[p.get('task_id', pid)] = sid
        a, b = p.get('candidate_a') or p.get('left'), p.get('candidate_b') or p.get('right')
        if a and b:
            cand_pair_to_id.setdefault((sid, _pair_key(a, b)), pid)

    by_scene: dict[str, dict] = defaultdict(dict)
    for row in supervision_rows:
        pid = row['pair_id']
        edge = prefer_edge((row['y_A'], row['y_T'], row['y_B']))
        if not edge:
            continue
        a, b = row.get('candidate_a'), row.get('candidate_b')
        if not a or not b:
            continue
        sid = row.get('scene_id') or scene_of.get(pid, '')
        by_scene[sid][(a, b)] = edge
        cand_pair_to_id.setdefault((sid, _pair_key(a, b)), pid)

    cycle_scenes = []
    edges_by_scene: dict[str, set[frozenset]] = defaultdict(set)
    for sid, edges in by_scene.items():
        if not sid:
            continue
        cycles = detect_cycles(edges)
        if not cycles:
            continue
        cycle_scenes.append({'scene_id': sid, 'n_cycles': len(cycles), 'cycles': cycles[:5]})
        for a, b, c in cycles:
            # Only the three undirected edges of each 3-cycle go to the queue.
            edges_by_scene[sid].update([_pair_key(a, b), _pair_key(b, c), _pair_key(c, a)])

    # Prefer denser cycles when capping the queue.
    cycle_scenes.sort(key=lambda s: -s['n_cycles'])
    cycle_pairs: list[str] = []
    seen_edge: set[tuple[str, frozenset]] = set()
    kept_scenes: list[dict] = []
    for sc in cycle_scenes:
        sid = sc['scene_id']
        hit_cap = False
        for ek in edges_by_scene.get(sid, ()):
            key = (sid, ek)
            if key in seen_edge:
                continue
            pid = cand_pair_to_id.get(key)
            if not pid:
                continue
            if len(cycle_pairs) >= max_queue:
                hit_cap = True
                break
            seen_edge.add(key)
            cycle_pairs.append(pid)
        kept_scenes.append(sc)
        if hit_cap:
            break
    cycle_pairs = list(dict.fromkeys(cycle_pairs))[:max_queue]

    rs = [float(r['R']) for r in supervision_rows if 'R' in r]
    r_varies = bool(rs) and (max(rs) - min(rs) > 1e-3)
    low_pairs: list[str] = []
    if r_varies:
        ranked = sorted(
            ((float(r['R']), r['pair_id']) for r in supervision_rows if float(r['R']) < r_thresh),
            key=lambda x: x[0],
        )
        low_pairs = [pid for _, pid in ranked[:low_r_budget]]

    reasons = []
    chosen: list[str] = []
    if cycle_pairs:
        reasons.append('cycle')
        chosen.extend(cycle_pairs)
    if low_pairs and len(chosen) < max_queue:
        extra = [pid for pid in low_pairs if pid not in set(chosen)]
        room = max_queue - len(chosen)
        if extra:
            take = room if not cycle_pairs else min(room, max(10, room // 2))
            if take > 0:
                chosen.extend(extra[:take])
                reasons.append('low_R')
    chosen = list(dict.fromkeys(chosen))

    if not chosen:
        ids = [(p.get('pair_id') or p['task_id']) for p in pairs]
        n = max(1, min(max_queue, int(round(len(ids) * frac)))) if ids else 0
        chosen = sample_reannotation_pairs(ids, frac=n / max(len(ids), 1), seed=seed)[:max_queue]
        reasons.append('random')

    n_low_in = len([p for p in chosen if p in set(low_pairs)]) if low_pairs else 0
    note = {
        'cycle': (
            f'Cycle-edge pairs are reviewed first '
            f'({len(kept_scenes)} scenes, {len(cycle_pairs)} edges; not whole scenes)'
        ),
        'low_R': f'Also added {n_low_in} pairs with the lowest current R (not every pair with R<{r_thresh})',
        'random': (
            f'No cycles yet, and R is nearly the same across pairs. '
            f'Sampling about {int(frac*100)}% at random (cap {max_queue})'
        ),
    }
    return {
        'pair_ids': chosen,
        'n': len(chosen),
        'reasons': reasons,
        'cycle_scenes': kept_scenes,
        'n_cycle_edges': len(cycle_pairs),
        'n_low_R': n_low_in,
        'note': '；'.join(note[r] for r in reasons if r in note),
    }



def quality_report(
    labels: Sequence[Label],
    labels_re: Sequence[Label] | None = None,
    labels_swap: Sequence[Label] | None = None,
    cycle_count: int = 0,
    n_pairs: int = 0,
    control_pass: float | None = None,
) -> dict:
    """§5.2 reliability table skeleton."""
    report = {
        'n_labels': len(labels),
        'tie_ratio': tie_ratio(labels),
        'cycle_rate': (cycle_count / n_pairs) if n_pairs else float('nan'),
        'control_pair_pass_rate': control_pass,
    }
    if labels_re is not None:
        report['reannotation_agreement'] = agreement_rate(labels, labels_re)
        report['cohens_kappa'] = cohens_kappa(labels, labels_re)
    if labels_swap is not None:
        report['candidate_swap_consistency'] = swap_consistency(labels, labels_swap)
    return report

"""Preference distribution y and optional reliability R (score.md §3.1.4–3.1.5).

Paper method: pool all annotations on a pair into empirical y_c = n_c / Σ n.
R / expertise weights remain available for QC and ablations, not for training.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from pref.schema import (
    LABEL_TO_INDEX,
    Expertise,
    Label,
)

DIRICHLET_ALPHA = np.ones(3, dtype=np.float64)  # α_A=α_T=α_B=1


@dataclass(frozen=True)
class PairSupervision:
    """Pair-level training target. Training uses y; R is optional metadata."""

    pair_id: str
    y: tuple[float, float, float]  # (y_A, y_T, y_B)
    R: float
    n_annotators: int
    n_annotations: int
    per_annotator_counts: dict[str, tuple[int, int, int]]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.y, dtype=np.float64)


def annotator_weight(expertise: Expertise, eta: float = 0.5) -> float:
    if not (0.0 < eta < 1.0):
        raise ValueError(f'eta must be in (0,1), got {eta}')
    return 1.0 if expertise == 'professional' else float(eta)


def count_labels(labels: Sequence[Label]) -> np.ndarray:
    """Return counts [n_A, n_T, n_B]."""
    n = np.zeros(3, dtype=np.float64)
    for lab in labels:
        n[LABEL_TO_INDEX[lab]] += 1.0
    return n


def empirical_pref(counts: np.ndarray) -> np.ndarray:
    total = float(counts.sum())
    if total <= 0:
        return np.ones(3, dtype=np.float64) / 3.0
    return counts / total


def dirichlet_var(counts: np.ndarray, alpha: np.ndarray = DIRICHLET_ALPHA) -> np.ndarray:
    """Var[θ_c | D] for each class under Dirichlet(n+α)."""
    a = counts + alpha
    a0 = float(a.sum())
    return a * (a0 - a) / (a0 ** 2 * (a0 + 1.0))


def aggregate_pair(
    labels_by_annotator: Mapping[str, Sequence[Label]],
    expertise: Mapping[str, Expertise] | None = None,
    eta: float = 0.5,
    alpha: np.ndarray | None = None,
    *,
    pool_annotations: bool = True,
) -> tuple[np.ndarray, float, dict]:
    """Aggregate labels into (y, R).

    Default ``pool_annotations=True`` (paper):
      y_c = n_c / Σ n_c over all annotations (no expertise weight).

    ``pool_annotations=False``: optional per-annotator λ-weighted form (analysis).
    """
    if not labels_by_annotator:
        raise ValueError('labels_by_annotator is empty')

    alpha = DIRICHLET_ALPHA if alpha is None else np.asarray(alpha, dtype=np.float64)
    expertise = expertise or {}

    per_counts: dict[str, tuple[int, int, int]] = {}
    pooled = np.zeros(3, dtype=np.float64)
    for uid, labs in labels_by_annotator.items():
        c = count_labels(list(labs))
        per_counts[uid] = (int(c[0]), int(c[1]), int(c[2]))
        pooled += c
    n_ann = int(pooled.sum())
    if n_ann <= 0:
        raise ValueError('no labels')

    if pool_annotations:
        y = empirical_pref(pooled)
        V = dirichlet_var(pooled, alpha=alpha)
        U = float(V.sum())
        U_max = 1.0 / 6.0
        R = float(np.clip(1.0 - U / U_max, 0.0, 1.0))
        meta = {
            'lambda': {'__pooled__': 1.0},
            'U': U,
            'U_max': U_max,
            'V': V.tolist(),
            'per_annotator_counts': per_counts,
            'n_annotations': n_ann,
            'n_annotators': len(per_counts),
            'pool_annotations': True,
        }
        return y, R, meta

    annotators = sorted(labels_by_annotator.keys())
    weights = np.asarray(
        [annotator_weight(expertise.get(u, 'professional'), eta) for u in annotators],
        dtype=np.float64,
    )
    lam = weights / weights.sum()
    y = np.zeros(3, dtype=np.float64)
    V = np.zeros(3, dtype=np.float64)
    for u, lam_u in zip(annotators, lam):
        labs = list(labels_by_annotator[u])
        if not labs:
            continue
        counts = count_labels(labs)
        y += lam_u * empirical_pref(counts)
        V += (lam_u ** 2) * dirichlet_var(counts, alpha)
    U = float(V.sum())
    lam2 = float((lam ** 2).sum())
    U_max = lam2 / 6.0
    R = 0.0 if U_max <= 0 else float(np.clip(1.0 - U / U_max, 0.0, 1.0))
    y = y / max(float(y.sum()), 1e-12)
    meta = {
        'lambda': {u: float(l) for u, l in zip(annotators, lam)},
        'U': U,
        'U_max': U_max,
        'V': V.tolist(),
        'per_annotator_counts': per_counts,
        'n_annotations': n_ann,
        'n_annotators': len(per_counts),
        'pool_annotations': False,
    }
    return y, R, meta


def aggregate_pair_supervision(
    pair_id: str,
    labels_by_annotator: Mapping[str, Sequence[Label]],
    expertise: Mapping[str, Expertise] | None = None,
    eta: float = 0.5,
    *,
    pool_annotations: bool = True,
) -> PairSupervision:
    y, R, meta = aggregate_pair(
        labels_by_annotator, expertise=expertise, eta=eta,
        pool_annotations=pool_annotations,
    )
    return PairSupervision(
        pair_id=pair_id,
        y=(float(y[0]), float(y[1]), float(y[2])),
        R=R,
        n_annotators=int(meta['n_annotators']),
        n_annotations=int(meta['n_annotations']),
        per_annotator_counts=meta['per_annotator_counts'],
    )


def group_labels_by_pair(
    annotations: Iterable[tuple[str, str, Label]],
) -> dict[str, dict[str, list[Label]]]:
    """annotations: iterable of (pair_id, annotator_id, label)."""
    out: dict[str, dict[str, list[Label]]] = defaultdict(lambda: defaultdict(list))
    for pair_id, ann_id, lab in annotations:
        out[pair_id][ann_id].append(lab)
    return {pid: dict(by_u) for pid, by_u in out.items()}


def aggregate_all_pairs(
    annotations: Iterable[tuple[str, str, Label]],
    expertise: Mapping[str, Expertise] | None = None,
    eta: float = 0.5,
    *,
    pool_annotations: bool = True,
) -> dict[str, PairSupervision]:
    grouped = group_labels_by_pair(annotations)
    return {
        pid: aggregate_pair_supervision(
            pid, by_u, expertise=expertise, eta=eta,
            pool_annotations=pool_annotations,
        )
        for pid, by_u in grouped.items()
    }

"""Canonical entities for the preference database (score.md §3.1.3).

Existing pref_bench JSONL uses display-side `left/right/tie` and method names.
This module maps those records onto fixed candidate identities A/B and labels
`A / Tie / B` without rewriting historical files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Expertise = Literal['professional', 'non-professional']
Response = Literal['left', 'tie', 'right']
Label = Literal['A', 'Tie', 'B']

LABEL_A: Label = 'A'
LABEL_TIE: Label = 'Tie'
LABEL_B: Label = 'B'
LABEL_TO_INDEX = {LABEL_A: 0, LABEL_TIE: 1, LABEL_B: 2}


@dataclass(frozen=True)
class Annotator:
    annotator_id: str
    expertise: Expertise = 'professional'


@dataclass
class Annotation:
    """One independent human judgment (append-only)."""

    annotation_id: str
    pair_id: str
    annotator_id: str
    round_id: int
    left_candidate: str
    right_candidate: str
    response: Response
    label: Label
    ts: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop('extra', {})
        d.update(extra)
        return d


def response_to_label(
    response: Response,
    left_candidate: str,
    right_candidate: str,
    candidate_a: str,
    candidate_b: str,
) -> Label:
    """Map UI response (left/tie/right) onto fixed pair identities A/B."""
    if response == 'tie':
        return LABEL_TIE
    if left_candidate == candidate_a and right_candidate == candidate_b:
        return LABEL_A if response == 'left' else LABEL_B
    if left_candidate == candidate_b and right_candidate == candidate_a:
        return LABEL_B if response == 'left' else LABEL_A
    raise ValueError(
        f'candidates mismatch: left={left_candidate}, right={right_candidate}, '
        f'A={candidate_a}, B={candidate_b}'
    )


def pair_fixed_ab(pair: dict) -> tuple[str, str]:
    """Canonical A/B for a pairs.jsonl row (lexicographic on method names)."""
    left = pair['left']
    right = pair['right']
    if left <= right:
        return left, right
    return right, left


def label_from_legacy_row(pair: dict, row: dict) -> Label:
    """Convert a legacy labels_*.jsonl row into A/Tie/B."""
    cand_a, cand_b = pair_fixed_ab(pair)
    left = row.get('left', pair['left'])
    right = row.get('right', pair['right'])
    choice = row['choice']
    if choice not in ('left', 'right', 'tie'):
        raise ValueError(f'unknown choice: {choice}')
    return response_to_label(choice, left, right, cand_a, cand_b)

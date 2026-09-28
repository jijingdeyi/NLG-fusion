"""Human-preference fusion quality learning (score.md §3–§4)."""

from pref.aggregate import aggregate_pair, aggregate_all_pairs
from pref.schema import LABEL_A, LABEL_B, LABEL_TIE, response_to_label

__all__ = [
    'LABEL_A',
    'LABEL_B',
    'LABEL_TIE',
    'response_to_label',
    'aggregate_pair',
    'aggregate_all_pairs',
]

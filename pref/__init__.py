"""Human-preference fusion quality learning."""

from pref.aggregate import aggregate_all_pairs, aggregate_pair
from pref.schema import LABEL_A, LABEL_B, LABEL_TIE, response_to_label

__all__ = [
    'LABEL_A',
    'LABEL_B',
    'LABEL_TIE',
    'response_to_label',
    'aggregate_pair',
    'aggregate_all_pairs',
]


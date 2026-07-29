"""DTRC-Fusion: degradation-token routed correction in front of a frozen fusion backbone."""

from dtrc.corrector import DualModalityCorrector
from dtrc.vocab import IR_MVP_EXPERTS, VI_MVP_EXPERTS

__all__ = [
    'DualModalityCorrector',
    'VI_MVP_EXPERTS',
    'IR_MVP_EXPERTS',
]

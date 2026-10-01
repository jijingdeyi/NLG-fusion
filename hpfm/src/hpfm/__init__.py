"""Human-Preference Fusion Metric (HPFM) for IVIF.

Install::

    pip install hpfm

Score a fusion (higher is better)::

    from hpfm import hpfm
    score = hpfm(ir, vis, fused)
"""

from __future__ import annotations

from hpfm._api import hpfm, hpfm_prefer, get_device, set_device

__all__ = ["hpfm", "hpfm_prefer", "get_device", "set_device", "__version__"]
__version__ = "0.1.0"

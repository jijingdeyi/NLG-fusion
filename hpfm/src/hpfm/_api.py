"""Public scoring API."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from hpfm._model import ResNet18ScalarScorer
from hpfm._preprocess import as_gray01

PathOrArray = Union[str, Path, np.ndarray]

_INPUT_SIZE = 224
_MODEL: ResNet18ScalarScorer | None = None
_DEVICE: str | None = None


def _weights_file() -> Path:
    return Path(__file__).resolve().with_name("weights.pt")


def get_device() -> str:
    if _DEVICE is not None:
        return _DEVICE
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def set_device(device: str) -> None:
    """Force device for subsequent calls (reloads weights if changed)."""
    global _MODEL, _DEVICE
    if _DEVICE != device:
        _MODEL = None
        _DEVICE = device


def _load_model(device: str | None = None) -> ResNet18ScalarScorer:
    global _MODEL, _DEVICE
    if device is None:
        device = get_device()
    if _MODEL is not None and _DEVICE == device:
        return _MODEL
    path = _weights_file()
    if not path.is_file():
        raise FileNotFoundError(
            f"HPFM weights missing: {path}\n"
            "Reinstall the package or run hpfm/scripts/sync_weights.py"
        )
    payload = torch.load(path, map_location=device, weights_only=False)
    model = ResNet18ScalarScorer().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    _MODEL = model
    _DEVICE = device
    return model


@torch.no_grad()
def hpfm(
    ir: PathOrArray,
    vis: PathOrArray,
    fused: PathOrArray,
    *,
    device: str | None = None,
) -> float:
    """Human-preference fusion score (higher = better).

    Parameters
    ----------
    ir, vis, fused
        File paths or HxW arrays (``uint8`` / float in ``[0, 255]`` or ``[0, 1]``).
    device
        Optional torch device; default CUDA if available else CPU.
    """
    model = _load_model(device)
    x = np.stack(
        [
            as_gray01(ir, _INPUT_SIZE),
            as_gray01(vis, _INPUT_SIZE),
            as_gray01(fused, _INPUT_SIZE),
        ],
        axis=0,
    )
    t = torch.from_numpy(x).unsqueeze(0).to(_DEVICE)
    return float(model(t).item())


@torch.no_grad()
def hpfm_prefer(
    ir: PathOrArray,
    vis: PathOrArray,
    fused_a: PathOrArray,
    fused_b: PathOrArray,
    *,
    device: str | None = None,
    tau: float = 1.0,
) -> dict[str, float]:
    """Pairwise preference between two fusions of the same IR/VIS pair.

    Returns ``s_a``, ``s_b``, ``delta``, and ``p_a`` = sigmoid(delta / tau).
    """
    s_a = hpfm(ir, vis, fused_a, device=device)
    s_b = hpfm(ir, vis, fused_b, device=device)
    delta = s_a - s_b
    p_a = float(1.0 / (1.0 + np.exp(-delta / max(tau, 1e-8))))
    return {"s_a": s_a, "s_b": s_b, "delta": delta, "p_a": p_a}

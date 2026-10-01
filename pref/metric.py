"""Drop-in preference metric: call like a classical IVIF indicator.

Example (paths)::

    from pref.metric import preference_score
    s = preference_score('ir.png', 'vi.png', 'fused.png')

Example (arrays, same convention as ``compute_fusion_metrics``)::

    s = preference_score(ir, vi, fused)  # HxW float/uint8 in [0,255] or [0,1]

The Stage-1 ResNet-18 checkpoint is loaded once (lazy) from
``checkpoints/scorer_stage1_best.pt`` unless you pass ``ckpt=`` / use
``PreferenceScorer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import cv2
import numpy as np
import torch

from pref.model.resnet_scorer import ResNet18ScalarScorer

PathOrArray = Union[str, Path, np.ndarray]

_DEFAULT_CKPT = (
    Path(__file__).resolve().parents[1] / 'checkpoints' / 'scorer_stage1_best.pt'
)
_GLOBAL = None  # PreferenceScorer | None


def default_checkpoint() -> Path:
    return _DEFAULT_CKPT


def _as_gray01(img: PathOrArray, size: int) -> np.ndarray:
    """Return float32 HxW in [0, 1], resized to ``size``."""
    if isinstance(img, (str, Path)):
        arr = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(img)
        arr = arr.astype(np.float32)
    else:
        arr = np.asarray(img)
        if arr.ndim == 3:
            # (H,W,C) or (C,H,W)
            if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = arr[0]
            else:
                arr = arr[..., 0] if arr.shape[-1] in (1, 3) else arr.mean(axis=-1)
        arr = np.asarray(arr, dtype=np.float32)
        if float(np.nanmax(arr)) > 1.5:
            pass  # [0,255]
        else:
            arr = arr * 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 255.0)
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr / 255.0


def _triple_tensor(
    ir: PathOrArray,
    vi: PathOrArray,
    fused: PathOrArray,
    *,
    size: int,
) -> torch.Tensor:
    x = np.stack(
        [_as_gray01(ir, size), _as_gray01(vi, size), _as_gray01(fused, size)],
        axis=0,
    )
    return torch.from_numpy(x)


class PreferenceScorer:
    """Reusable handle: load weights once, score many fusions."""

    def __init__(
        self,
        ckpt: Path | str | None = None,
        *,
        device: str | None = None,
        size: int | None = None,
    ) -> None:
        path = Path(ckpt) if ckpt is not None else default_checkpoint()
        if not path.is_file():
            raise FileNotFoundError(
                f'checkpoint not found: {path}\n'
                f'Place scorer_stage1_best.pt under checkpoints/ or pass ckpt=...'
            )
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        blob = torch.load(path, map_location=device, weights_only=False)
        model = ResNet18ScalarScorer(pretrained=False).to(device)
        model.load_state_dict(blob['model'])
        model.eval()
        self.model = model
        self.device = device
        self.size = int(size if size is not None else blob.get('size', 224))
        self.ckpt = path
        self.meta = {k: blob[k] for k in blob if k != 'model'}

    @torch.no_grad()
    def __call__(
        self,
        ir: PathOrArray,
        vi: PathOrArray,
        fused: PathOrArray,
    ) -> float:
        """Scalar preference score for one fusion (higher = better)."""
        x = _triple_tensor(ir, vi, fused, size=self.size).unsqueeze(0).to(self.device)
        return float(self.model(x).item())

    @torch.no_grad()
    def batch(
        self,
        triples: Sequence[tuple[PathOrArray, PathOrArray, PathOrArray]],
        *,
        batch_size: int = 16,
    ) -> list[float]:
        """Score many (ir, vi, fused) triples; returns list of floats."""
        if not triples:
            return []
        out: list[float] = []
        for i in range(0, len(triples), batch_size):
            chunk = triples[i : i + batch_size]
            xs = torch.stack(
                [_triple_tensor(a, b, c, size=self.size) for a, b, c in chunk]
            ).to(self.device)
            out.extend(float(v) for v in self.model(xs).detach().cpu().tolist())
        return out

    @torch.no_grad()
    def prefer(
        self,
        ir: PathOrArray,
        vi: PathOrArray,
        fused_a: PathOrArray,
        fused_b: PathOrArray,
        *,
        tau: float = 1.0,
    ) -> dict[str, float]:
        """Pairwise preference between two fusions of the same IR/VI pair.

        Returns ``s_a``, ``s_b``, ``delta`` (= s_a - s_b), and
        ``p_a`` = sigmoid(delta / tau) ≈ P(A preferred over B).
        """
        s_a = self(ir, vi, fused_a)
        s_b = self(ir, vi, fused_b)
        delta = s_a - s_b
        p_a = float(1.0 / (1.0 + np.exp(-delta / max(tau, 1e-8))))
        return {'s_a': s_a, 's_b': s_b, 'delta': delta, 'p_a': p_a}


def get_scorer(
    ckpt: Path | str | None = None,
    *,
    device: str | None = None,
    size: int | None = None,
    reload: bool = False,
) -> PreferenceScorer:
    """Process-wide singleton (same idea as loading EN once)."""
    global _GLOBAL
    path = Path(ckpt) if ckpt is not None else default_checkpoint()
    need = (
        reload
        or _GLOBAL is None
        or Path(_GLOBAL.ckpt) != path
        or (device is not None and _GLOBAL.device != device)
        or (size is not None and _GLOBAL.size != size)
    )
    if need:
        _GLOBAL = PreferenceScorer(path, device=device, size=size)
    return _GLOBAL


def preference_score(
    ir: PathOrArray,
    vi: PathOrArray,
    fused: PathOrArray,
    *,
    ckpt: Path | str | None = None,
    device: str | None = None,
    size: int | None = None,
) -> float:
    """One-call preference metric (lazy-loads the default checkpoint).

    Drop-in style matching classical metrics::

        s = preference_score(ir, vi, fused)
    """
    return get_scorer(ckpt, device=device, size=size)(ir, vi, fused)


def preference_prefer(
    ir: PathOrArray,
    vi: PathOrArray,
    fused_a: PathOrArray,
    fused_b: PathOrArray,
    *,
    ckpt: Path | str | None = None,
    device: str | None = None,
    tau: float = 1.0,
) -> dict[str, float]:
    """One-call pairwise preference between two fusions."""
    return get_scorer(ckpt, device=device).prefer(
        ir, vi, fused_a, fused_b, tau=tau,
    )

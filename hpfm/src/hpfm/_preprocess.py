"""Image loading / resize helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

PathOrArray = Union[str, Path, np.ndarray]


def _read_gray_u8(path: Path) -> np.ndarray:
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img
    except ImportError:
        from PIL import Image

        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _resize(arr: np.ndarray, size: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        return np.asarray(
            Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.BILINEAR),
            dtype=np.float32,
        )


def as_gray01(img: PathOrArray, size: int) -> np.ndarray:
    """Return float32 HxW in [0, 1], resized to ``size``."""
    if isinstance(img, (str, Path)):
        arr = _read_gray_u8(Path(img)).astype(np.float32)
    else:
        arr = np.asarray(img)
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = arr[0]
            else:
                arr = arr[..., 0] if arr.shape[-1] in (1, 3) else arr.mean(axis=-1)
        arr = np.asarray(arr, dtype=np.float32)
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
    arr = np.nan_to_num(np.clip(arr, 0.0, 255.0), nan=0.0)
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = _resize(arr, size)
    return arr.astype(np.float32) / 255.0

"""IVIF validation metrics and configurable max-min competitive score."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from kornia.metrics import ssim as kornia_ssim
from numpy.typing import NDArray
from scipy.signal import convolve2d

from ldt.qabf import get_Qabf

# All raw metrics we may compute / normalize.
ALL_METRIC_KEYS = ('mi', 'qabf', 'scd', 'vif', 'ssim', 'sf')

# §7.8 / §8 recipes: which keys enter max-min.
SCORE_RECIPES: dict[str, tuple[str, ...]] = {
    'A': ('mi', 'qabf', 'scd', 'vif', 'ssim'),       # status quo
    'B': ('mi', 'qabf', 'scd', 'ssim'),              # drop VIF
    'C': ('mi', 'qabf', 'scd', 'vif'),               # drop SSIM
    'D': ('mi', 'qabf', 'sf'),                       # info + edge transfer + clarity
    'E': ('mi', 'qabf', 'sf', 'scd'),                # D + SCD
    'F': ('mi', 'qabf', 'sf', 'vif'),                # D + VIF
    'G': ('mi', 'vif', 'sf'),                        # info + fidelity + clarity (no Qabf)
}

DEFAULT_SCORE_RECIPE = 'A'

# Legacy alias: keys of recipe A.
METRIC_KEYS = SCORE_RECIPES[DEFAULT_SCORE_RECIPE]

# LLVIP.xlsx sheet ``mean`` row → floor=median, target=max across methods.
COMPETITIVE_RANGES: dict[str, tuple[float, float]] = {
    'mi': (2.045000, 3.133000),
    'qabf': (0.554000, 0.698000),
    'scd': (1.455000, 1.748000),
    'vif': (0.760000, 0.978000),
    'ssim': (0.663000, 0.691000),
    'sf': (13.562000, 22.863000),
}

XLSX_SHEET_MAP = {
    'mi': 'MI',
    'qabf': 'Qabf',
    'scd': 'SCD',
    'vif': 'VIF',
    'ssim': 'SSIM',
    'sf': 'SF',
}

# Legacy wide ranges (old equal-weight score). Kept only for reference.
VAL_METRIC_RANGES = {
    'mi': (1.5, 6.5),
    'scd': (1.0, 1.9),
    'vif': (0.4, 1.2),
    'ssim': (0.5, 0.85),
    'qabf': (0.1, 0.8),
    'sf': (5.0, 25.0),
}

_WORST_WEIGHT = 0.95
_MEAN_WEIGHT = 0.05


def resolve_score_config(
    recipe: str | None = None,
    metric_keys: Sequence[str] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[tuple[str, ...], dict[str, tuple[float, float]]]:
    """Resolve (metric_keys, ranges) for max-min scoring.

    Priority: explicit ``metric_keys`` > ``recipe`` > ``DEFAULT_SCORE_RECIPE``.
    """
    if metric_keys is not None:
        keys = tuple(metric_keys)
    else:
        name = recipe if recipe is not None else DEFAULT_SCORE_RECIPE
        if name not in SCORE_RECIPES:
            raise KeyError(
                f'Unknown score recipe {name!r}; expected one of {sorted(SCORE_RECIPES)}',
            )
        keys = SCORE_RECIPES[name]

    base = ranges or COMPETITIVE_RANGES
    resolved: dict[str, tuple[float, float]] = {}
    for key in keys:
        if key not in base:
            raise KeyError(f'No competitive range for metric {key!r}')
        resolved[key] = (float(base[key][0]), float(base[key][1]))
    return keys, resolved


_SUMMARY_ROW_LABELS = frozenset({'mean', 'std', 'avg', 'average', 'median'})


def load_method_means_from_xlsx(
    path: str,
    sheets: dict[str, str] | None = None,
    metric_keys: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Read per-method means from a metric workbook.

    Prefer the sheet's existing ``mean`` summary row (tables already store it).
    Only fall back to averaging per-image rows if ``mean`` is missing.
    Returns ``(method_names, {method: {metric_key: mean}})``.
    """
    import openpyxl

    keys = tuple(metric_keys) if metric_keys is not None else ALL_METRIC_KEYS
    sheet_map = sheets or XLSX_SHEET_MAP
    wb = openpyxl.load_workbook(path, data_only=True)

    method_names: list[str] | None = None
    per_metric: dict[str, list[float]] = {}

    for key in keys:
        if key not in sheet_map:
            raise KeyError(f'No sheet mapping for metric {key!r}')
        sheet_name = sheet_map[key]
        if sheet_name not in wb.sheetnames:
            raise KeyError(f'Sheet {sheet_name!r} missing in {path}')
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            raise ValueError(f'Empty sheet {sheet_name!r} in {path}')
        header = rows[0]
        names = [str(h) for h in header[1:] if h is not None]
        if method_names is None:
            method_names = names
        elif names != method_names:
            raise ValueError(
                f'Method columns differ on sheet {sheet_name!r} in {path}: '
                f'{names} vs {method_names}',
            )
        n_methods = len(method_names)

        mean_row = next(
            (
                r for r in rows[1:]
                if r[0] is not None and str(r[0]).strip().lower() == 'mean'
            ),
            None,
        )
        if mean_row is not None:
            means = []
            for j in range(n_methods):
                v = mean_row[j + 1]
                if v is None or not np.isfinite(float(v)):
                    raise ValueError(
                        f'Bad mean cell for {method_names[j]!r} / {key!r} in {path}',
                    )
                means.append(float(v))
        else:
            means = []
            for j in range(n_methods):
                col = []
                for r in rows[1:]:
                    if r[0] is None:
                        continue
                    if str(r[0]).strip().lower() in _SUMMARY_ROW_LABELS:
                        continue
                    v = r[j + 1]
                    if v is not None and np.isfinite(float(v)):
                        col.append(float(v))
                if not col:
                    raise ValueError(
                        f'No values for method {method_names[j]!r} / metric {key!r} in {path}',
                    )
                means.append(float(np.mean(col)))
        per_metric[key] = means

    assert method_names is not None
    out: dict[str, dict[str, float]] = {
        name: {key: per_metric[key][i] for key in keys}
        for i, name in enumerate(method_names)
    }
    return method_names, out


def build_competitive_ranges_from_means(
    method_means: Mapping[str, Mapping[str, float]],
    metric_keys: Sequence[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """(floor, target) = (median, max) over methods for each metric."""
    keys = tuple(metric_keys) if metric_keys is not None else ALL_METRIC_KEYS
    out: dict[str, tuple[float, float]] = {}
    for key in keys:
        arr = np.asarray(
            [float(method_means[m][key]) for m in method_means if key in method_means[m]],
            dtype=np.float64,
        )
        if arr.size == 0:
            raise ValueError(f'No values for metric {key!r}')
        floor = float(np.median(arr))
        target = float(np.max(arr))
        if target <= floor:
            target = floor + 1e-6
        out[key] = (floor, target)
    return out


def build_competitive_ranges_from_xlsx(
    path: str,
    sheets: dict[str, str] | None = None,
    metric_keys: Sequence[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Derive (floor, target) = (median, max) of per-method means from a metric workbook."""
    _, means = load_method_means_from_xlsx(path, sheets=sheets, metric_keys=metric_keys)
    return build_competitive_ranges_from_means(means, metric_keys=metric_keys)


def metric_attainment(
    value: float,
    floor: float,
    target: float,
) -> float:
    """Map raw metric to [-1, 1]: 0=median competitor, 1=best competitor (capped)."""
    if not np.isfinite(value) or target <= floor:
        return -1.0
    return float(np.clip((float(value) - floor) / (target - floor), -1.0, 1.0))


def score_from_metrics(
    metrics: Mapping[str, float],
    *,
    recipe: str | None = None,
    metric_keys: Sequence[str] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[float, dict[str, float | str]]:
    """Max-min competitive score from a metric dict (prefer dataset means)."""
    keys, rng = resolve_score_config(recipe=recipe, metric_keys=metric_keys, ranges=ranges)
    missing = [k for k in keys if k not in metrics]
    if missing:
        raise KeyError(f'Metrics dict missing keys required by score config: {missing}')

    attainment = {
        key: metric_attainment(float(metrics[key]), *rng[key])
        for key in keys
    }
    arr = np.asarray([attainment[k] for k in keys], dtype=np.float64)
    worst_i = int(np.argmin(arr))
    raw_score = _WORST_WEIGHT * float(arr.min()) + _MEAN_WEIGHT * float(arr.mean())
    score = float(np.clip((raw_score + 1.0) * 0.5, 0.0, 1.0))
    details: dict[str, float | str] = {
        **{f'a_{k}': float(v) for k, v in attainment.items()},
        'worst_metric': keys[worst_i],
        'worst_attainment': float(arr[worst_i]),
        'raw_score': raw_score,
        'recipe_keys': ','.join(keys),
    }
    return score, details


def composite_validation_score(
    mi: float | None = None,
    qabf: float | None = None,
    scd: float | None = None,
    vif: float | None = None,
    ssim_score: float | None = None,
    *,
    sf: float | None = None,
    metrics: Mapping[str, float] | None = None,
    recipe: str | None = None,
    metric_keys: Sequence[str] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> float:
    """Max-min score in [0, 1]. Prefer ``metrics=...`` or ``score_from_metrics``."""
    score, _ = composite_validation_score_with_details(
        mi, qabf, scd, vif, ssim_score,
        sf=sf, metrics=metrics, recipe=recipe, metric_keys=metric_keys, ranges=ranges,
    )
    return score


def composite_validation_score_with_details(
    mi: float | None = None,
    qabf: float | None = None,
    scd: float | None = None,
    vif: float | None = None,
    ssim_score: float | None = None,
    *,
    sf: float | None = None,
    metrics: Mapping[str, float] | None = None,
    recipe: str | None = None,
    metric_keys: Sequence[str] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[float, dict[str, float | str]]:
    if metrics is None:
        metrics = {}
        if mi is not None:
            metrics['mi'] = mi
        if qabf is not None:
            metrics['qabf'] = qabf
        if scd is not None:
            metrics['scd'] = scd
        if vif is not None:
            metrics['vif'] = vif
        if ssim_score is not None:
            metrics['ssim'] = ssim_score
        if sf is not None:
            metrics['sf'] = sf
    return score_from_metrics(
        metrics, recipe=recipe, metric_keys=metric_keys, ranges=ranges,
    )


def spatial_frequency(image: np.ndarray) -> float:
    """Spatial frequency (SF) on a 2D array (typically [0, 255])."""
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f'spatial_frequency expects 2D array, got shape {x.shape}')
    rf = np.sqrt(np.mean(np.diff(x, axis=1) ** 2))
    cf = np.sqrt(np.mean(np.diff(x, axis=0) ** 2))
    return float(np.sqrt(rf * rf + cf * cf))


def fspecial_gaussian(shape: tuple[int, int], sigma: float) -> NDArray[np.float64]:
    m, n = [(ss - 1) // 2 for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x.astype(np.float64) ** 2 + y.astype(np.float64) ** 2) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0.0
    return h / h.sum()


def vifp_mscale(ref, dist):
    sigma_nsq = 2
    num = den = 0
    for scale in range(1, 5):
        n = 2 ** (4 - scale + 1) + 1
        win = fspecial_gaussian((n, n), n / 5)
        if scale > 1:
            ref = convolve2d(ref, win, mode='valid')
            dist = convolve2d(dist, win, mode='valid')
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]
        mu1 = convolve2d(ref, win, mode='valid')
        mu2 = convolve2d(dist, win, mode='valid')
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        sigma1_sq = convolve2d(ref * ref, win, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(dist * dist, win, mode='valid') - mu2_sq
        sigma12 = convolve2d(ref * dist, win, mode='valid') - mu1_mu2
        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0
        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12
        g[sigma1_sq < 1e-10] = 0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq < 1e-10] = 0
        g[sigma2_sq < 1e-10] = 0
        sv_sq[sigma2_sq < 1e-10] = 0
        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= 1e-10] = 1e-10
        num += np.sum(np.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
    if den == 0 or not np.isfinite(den) or not np.isfinite(num):
        return 0.0
    return float(num / den)


def _safe_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt(max(np.sum(xm * xm), 0.0) * max(np.sum(ym * ym), 0.0)) + eps
    return float(np.clip(np.sum(xm * ym) / denom, -1.0, 1.0))


def _hab(im1, im2, gray_level=256, use_nats=True):
    im1 = np.clip(np.asarray(im1, dtype=np.int32), 0, gray_level - 1).ravel()
    im2 = np.clip(np.asarray(im2, dtype=np.int32), 0, gray_level - 1).ravel()
    log_fn = np.log if use_nats else np.log2
    h, _, _ = np.histogram2d(
        im1, im2, bins=gray_level, range=[[0, gray_level], [0, gray_level]],
    )
    total = h.sum()
    if total <= 0:
        return 0.0
    h = h / total
    im1_marg, im2_marg = h.sum(axis=1), h.sum(axis=0)

    def _ent(p):
        p = p[p > 0]
        return float(np.sum(p * log_fn(p))) if p.size else 0.0

    return float(_ent(h.ravel()) - _ent(im1_marg) - _ent(im2_marg))


def compute_fusion_metrics(
    ir: np.ndarray,
    vi: np.ndarray,
    fused: np.ndarray,
    *,
    recipe: str | None = None,
    metric_keys: Sequence[str] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """ir/vi/fused: float32 arrays in [0, 255], shape (H, W).

    Always returns raw metrics in ``ALL_METRIC_KEYS``. ``score`` is per-image
    (debug only); for model selection, average raw metrics then call
    ``score_from_metrics`` once.
    """
    ir = np.nan_to_num(np.asarray(ir, dtype=np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    vi = np.nan_to_num(np.asarray(vi, dtype=np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    fused = np.nan_to_num(np.asarray(fused, dtype=np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    ir = np.clip(ir, 0.0, 255.0)
    vi = np.clip(vi, 0.0, 255.0)
    fused = np.clip(fused, 0.0, 255.0)

    mi = _hab(ir, fused) + _hab(vi, fused)
    qabf = get_Qabf(ir, vi, fused)
    scd = _safe_corr(fused - vi, ir) + _safe_corr(fused - ir, vi)
    vif = vifp_mscale(ir, fused) + vifp_mscale(vi, fused)
    sf = spatial_frequency(fused)

    ir_t = torch.tensor(ir / 255.0, dtype=torch.float32).view(1, 1, *ir.shape)
    vi_t = torch.tensor(vi / 255.0, dtype=torch.float32).view(1, 1, *vi.shape)
    f_t = torch.tensor(fused / 255.0, dtype=torch.float32).view(1, 1, *fused.shape)
    ssim_val = (
        kornia_ssim(ir_t, f_t, window_size=11, max_val=1.0).mean()
        + kornia_ssim(vi_t, f_t, window_size=11, max_val=1.0).mean()
    ).item() * 0.5

    def _finite(x: float, default: float = 0.0) -> float:
        x = float(x)
        return x if np.isfinite(x) else default

    raw = {
        'mi': _finite(mi),
        'qabf': _finite(qabf),
        'scd': _finite(scd),
        'vif': _finite(vif),
        'ssim': _finite(ssim_val),
        'sf': _finite(sf),
    }
    score, _ = score_from_metrics(
        raw, recipe=recipe, metric_keys=metric_keys, ranges=ranges,
    )
    return {**raw, 'score': score}

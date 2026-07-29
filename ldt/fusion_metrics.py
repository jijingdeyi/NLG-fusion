"""IVIF validation metrics and composite score (adapted from ca-fusion-loss)."""

from __future__ import annotations

import numpy as np
import torch
from kornia.metrics import ssim as kornia_ssim
from numpy.typing import NDArray
from scipy.signal import convolve2d

from ldt.qabf import get_Qabf

VAL_METRIC_RANGES = {
    'mi': (1.5, 6.5),
    'scd': (1.0, 1.9),
    'vif': (0.4, 1.2),
    'ssim': (0.5, 0.85),
    'qabf': (0.1, 0.8),
}


def composite_validation_score(
    mi: float, qabf: float, scd: float, vif: float, ssim_score: float,
) -> float:
    keys = ('mi', 'qabf', 'scd', 'vif', 'ssim')
    raw = (mi, qabf, scd, vif, ssim_score)
    normed = []
    for key, m in zip(keys, raw):
        lo, hi = VAL_METRIC_RANGES[key]
        if hi <= lo:
            normed.append(0.0)
            continue
        normed.append(float(np.clip((m - lo) / (hi - lo), 0.0, 1.0)))
    return float(sum(normed) / len(normed))


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


def compute_fusion_metrics(ir: np.ndarray, vi: np.ndarray, fused: np.ndarray) -> dict[str, float]:
    """ir/vi/fused: float32 arrays in [0, 255], shape (H, W)."""
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

    mi, qabf, scd, vif, ssim_val = map(_finite, (mi, qabf, scd, vif, ssim_val))
    score = composite_validation_score(mi, qabf, scd, vif, ssim_val)
    return {'mi': mi, 'qabf': qabf, 'scd': scd, 'vif': vif, 'ssim': ssim_val, 'score': score}

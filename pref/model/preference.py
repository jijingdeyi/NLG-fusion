"""A/Tie/B preference probabilities from (s, σ) (score.md §4.3)."""

from __future__ import annotations

import math

import torch

_SQRT2 = math.sqrt(2.0)


def standard_normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / _SQRT2))


def preference_probs(
    s_a: torch.Tensor,
    s_b: torch.Tensor,
    sigma_a: torch.Tensor,
    sigma_b: torch.Tensor,
    delta: float = 0.25,
    sigma0: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (p̂_A, p̂_T, p̂_B) under Gaussian latent quality + tie band [-δ, δ]."""
    ds = s_a - s_b
    sigma_ab = torch.sqrt(sigma_a ** 2 + sigma_b ** 2 + sigma0 ** 2).clamp_min(1e-6)
    p_a = standard_normal_cdf((ds - delta) / sigma_ab)
    p_b = standard_normal_cdf((-ds - delta) / sigma_ab)
    p_t = (1.0 - p_a - p_b).clamp_min(1e-8)
    z = p_a + p_t + p_b
    return p_a / z, p_t / z, p_b / z


def preference_distribution(
    s_a: torch.Tensor,
    s_b: torch.Tensor,
    sigma_a: torch.Tensor,
    sigma_b: torch.Tensor,
    delta: float = 0.25,
    sigma0: float = 0.1,
) -> torch.Tensor:
    p_a, p_t, p_b = preference_probs(s_a, s_b, sigma_a, sigma_b, delta=delta, sigma0=sigma0)
    return torch.stack([p_a, p_t, p_b], dim=-1)

"""Reliability-aware preference losses.

Stage 1: soft pairwise BCE on q_A = y_A + ½ y_T (scalar ranking).
Stage 2: full A/Tie/B CE with Gaussian preference model (score.md §4.4).
"""

from __future__ import annotations

import torch

from pref.model.preference import preference_distribution


def soft_preference_target(y: torch.Tensor) -> torch.Tensor:
    """q_A = y_A + ½ y_T  (Tie as half-vote toward indifference)."""
    return y[..., 0] + 0.5 * y[..., 1]


def rank_bce_loss(
    q_hat: torch.Tensor,
    q: torch.Tensor,
    R: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L_rank = mean_p [q log q̂ + (1-q) log(1-q̂)], optionally R-weighted."""
    q_hat = q_hat.clamp(eps, 1.0 - eps)
    per = -(q * torch.log(q_hat) + (1.0 - q) * torch.log(1.0 - q_hat))
    if R is None:
        return per.mean()
    w = R.clamp_min(0.0)
    denom = w.sum().clamp_min(eps)
    return (w * per).sum() / denom


def scorer_rank_loss(
    model: torch.nn.Module,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    y: torch.Tensor,
    R: torch.Tensor,
    tau: float = 1.0,
    swap_prob: float = 0.5,
    use_R: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Shared scalar scorer + soft pairwise BCE (paper method; use_R=False)."""
    if swap_prob > 0 and torch.rand(()) < swap_prob:
        x_a, x_b = x_b, x_a
        y = y[:, [2, 1, 0]]

    s_a = model(x_a)
    s_b = model(x_b)
    if isinstance(s_a, tuple):
        s_a, s_b = s_a[0], s_b[0]

    q = soft_preference_target(y)
    q_hat = torch.sigmoid((s_a - s_b) / float(tau))
    loss = rank_bce_loss(q_hat, q, R if use_R else None)

    with torch.no_grad():
        decisive = (q - 0.5).abs() >= 0.1
        if decisive.any():
            pred_a = q_hat[decisive] >= 0.5
            true_a = q[decisive] >= 0.5
            acc_dec = (pred_a == true_a).float().mean()
        else:
            acc_dec = q_hat.new_tensor(float('nan'))
        soft_acc = ((q_hat >= 0.5) == (q >= 0.5)).float().mean()

    stats = {
        'loss': float(loss.detach()),
        'acc_decisive': float(acc_dec),
        'acc_soft': float(soft_acc),
        's_mean': float(0.5 * (s_a.mean() + s_b.mean()).detach()),
        'ds_mean': float((s_a - s_b).mean().detach()),
        'q_mean': float(q.mean().detach()),
    }
    return loss, stats


def preference_ce_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    R: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """L_pref = - Σ_p R_p Σ_c y_{c,p} log p̂_{c,p} / Σ_p R_p."""
    log_p = torch.log(y_hat.clamp_min(eps))
    per = -(y * log_p).sum(dim=-1)  # (B,)
    w = R.clamp_min(0.0)
    denom = w.sum().clamp_min(eps)
    return (w * per).sum() / denom


def scorer_pair_loss(
    model: torch.nn.Module,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    y: torch.Tensor,
    R: torch.Tensor,
    delta: float = 0.25,
    sigma0: float = 0.1,
    swap_prob: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Stage-2: shared scorer (s,σ) + R-weighted A/Tie/B CE.

    With probability `swap_prob`, swap (A,B) and (y_A,y_T,y_B)→(y_B,y_T,y_A).
    """
    if swap_prob > 0 and torch.rand(()) < swap_prob:
        x_a, x_b = x_b, x_a
        y = y[:, [2, 1, 0]]

    s_a, sig_a = model(x_a)
    s_b, sig_b = model(x_b)
    y_hat = preference_distribution(s_a, s_b, sig_a, sig_b, delta=delta, sigma0=sigma0)
    loss = preference_ce_loss(y_hat, y, R)
    with torch.no_grad():
        pred = y_hat.argmax(dim=-1)
        hard = y.argmax(dim=-1)
        acc = (pred == hard).float().mean()
    stats = {
        'loss': float(loss.detach()),
        'acc_mode': float(acc),
        's_mean': float(0.5 * (s_a.mean() + s_b.mean()).detach()),
        'sigma_mean': float(0.5 * (sig_a.mean() + sig_b.mean()).detach()),
    }
    return loss, stats

"""Stage-1 soft ranking helpers."""

from __future__ import annotations

import torch

from pref.model.loss import rank_bce_loss, soft_preference_target


def test_soft_preference_target():
    y = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.7, 0.2, 0.1],
    ])
    q = soft_preference_target(y)
    assert torch.allclose(q, torch.tensor([1.0, 0.5, 0.8]))


def test_rank_bce_perfect():
    q = torch.tensor([1.0, 0.0])
    q_hat = torch.tensor([0.999, 0.001])
    R = torch.ones(2)
    loss = rank_bce_loss(q_hat, q, R)
    assert float(loss) < 0.02

    # Tie half-vote: q=0.5, q̂=0.5 → -log(0.5) ≈ 0.693
    loss_tie = rank_bce_loss(torch.tensor([0.5]), torch.tensor([0.5]), torch.ones(1))
    assert abs(float(loss_tie) - 0.693147) < 1e-3

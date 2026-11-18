"""Region scorer: one score per feature-map cell, plus a whole-image score."""
import torch

from pref.model.region_scorer import RegionScorer


def test_mix_starts_even_and_native_sizes_both_score():
    torch.manual_seed(4)
    model = RegionScorer(pretrained=False).eval()
    assert torch.equal(model.mix.detach().cpu(), torch.tensor([0.5, 0.5]))
    small = (
        torch.rand(1, 1, 40, 48),
        torch.rand(1, 3, 40, 48),
        torch.rand(1, 3, 40, 48),
    )
    large = (
        torch.rand(1, 1, 80, 64),
        torch.rand(1, 3, 80, 64),
        torch.rand(1, 3, 80, 64),
    )
    assert model(*small).shape == (1,)
    assert model(*large).shape == (1,)


def test_zero_chroma_ignores_recoloring_and_new_heads_train_while_resnet_is_fixed():
    torch.manual_seed(5)
    model = RegionScorer(pretrained=False).eval()
    ir = torch.rand(1, 1, 48, 40)
    vi = torch.rand(1, 3, 48, 40)
    fused = torch.rand(1, 3, 48, 40)
    recolored = fused.clone()
    recolored[:, 1:] = torch.rand_like(recolored[:, 1:])
    assert torch.allclose(model(ir, vi, fused), model(ir, vi, recolored), atol=1e-5)
    model.set_backbone_trainable(False)
    assert next(model.layer4.parameters()).requires_grad is False
    assert model.region_head[0].weight.requires_grad is True
    assert model.global_head[0].weight.requires_grad is True
    assert model.mix.requires_grad is True
    assert model.chroma.weight.requires_grad is True
    model.zero_grad(set_to_none=True)
    model(ir, vi, fused).sum().backward()
    assert model.region_head[-1].weight.grad.abs().sum() > 0
    assert float(model.chroma.weight.detach().abs().sum()) == 0.0


def test_corner_pixel_reaches_the_region_score():
    torch.manual_seed(6)
    model = RegionScorer(pretrained=False).eval()
    ir = torch.rand(1, 1, 36, 44)
    vi = torch.rand(1, 3, 36, 44)
    fused = torch.rand(1, 3, 36, 44, requires_grad=True)
    model(ir, vi, fused).sum().backward()
    assert fused.grad[..., 0, 0].abs().sum() > 0
    assert fused.grad[..., -1, -1].abs().sum() > 0

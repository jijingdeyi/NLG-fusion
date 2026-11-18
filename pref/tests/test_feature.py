"""Feature scorer: one weight per deep channel, shared across the image."""
import torch

from pref.model.feature_scorer import FeatureScorer


def test_channel_weights_are_uniform_and_not_a_spatial_map():
    model = FeatureScorer(pretrained=False).eval()
    assert model.channel_logit.shape == (512,)
    assert model.channel_logit.ndim == 1
    weight = torch.softmax(model.channel_logit.detach(), dim=0)
    assert torch.allclose(weight, torch.full((512,), 1.0 / 512))
    assert torch.equal(model.mix.detach().cpu(), torch.tensor([0.5, 0.5]))
    small = (torch.rand(1, 1, 40, 48), torch.rand(1, 3, 40, 48), torch.rand(1, 3, 40, 48))
    wide = (torch.rand(1, 1, 36, 80), torch.rand(1, 3, 36, 80), torch.rand(1, 3, 36, 80))
    assert model(*small).shape == (1,)
    assert model(*wide).shape == (1,)


def test_zero_chroma_ignores_recoloring_and_channel_weights_train():
    torch.manual_seed(7)
    model = FeatureScorer(pretrained=False).eval()
    ir = torch.rand(1, 1, 48, 40)
    vi = torch.rand(1, 3, 48, 40)
    fused = torch.rand(1, 3, 48, 40)
    recolored = fused.clone()
    recolored[:, 1:] = torch.rand_like(recolored[:, 1:])
    assert torch.allclose(model(ir, vi, fused), model(ir, vi, recolored), atol=1e-5)
    model.set_backbone_trainable(False)
    assert next(model.layer4.parameters()).requires_grad is False
    assert model.channel_logit.requires_grad is True
    assert model.channel_gain.requires_grad is True
    model.zero_grad(set_to_none=True)
    model(ir, vi, fused).sum().backward()
    assert model.channel_logit.grad.abs().sum() > 0
    assert float(model.chroma.weight.detach().abs().sum()) == 0.0

"""Baseline scorer: YCbCr channels and coverage-weighted tiles."""
import torch

from pref.model.baseline import BaselineScorer, coverage_weights, covers_image, tile_boxes


def test_tiles_cover_the_image_and_partition_its_area():
    for height, width, tile in ((80, 100, 224), (224, 224, 224), (448, 448, 224), (500, 600, 224)):
        assert covers_image(height, width, tile)
        weights = coverage_weights(height, width, tile)
        assert abs(float(weights.sum()) - height * width) < 1e-3
    boxes = tile_boxes(500, 600, 224)
    assert len(boxes) == 9
    assert (276, 376, 224, 224) in boxes
    wide = coverage_weights(224, 500, 224)
    assert wide.shape == (3,)
    assert float(wide[0]) > float(wide[1])
    assert torch.allclose(wide, coverage_weights(224, 500, 224))


def test_zero_chroma_matches_luma_and_still_receives_a_gradient():
    torch.manual_seed(8)
    model = BaselineScorer(pretrained=False, global_size=32, tile_size=32)
    ir = torch.rand(1, 1, 40, 36)
    vi = torch.rand(1, 3, 40, 36)
    fused = torch.rand(1, 3, 40, 36)
    recolored = fused.clone()
    recolored[:, 1:] = torch.rand_like(recolored[:, 1:])
    model.eval()
    assert model(ir, vi, fused).shape == (1,)
    assert torch.allclose(model(ir, vi, fused), model(ir, vi, recolored), atol=1e-5)
    assert torch.allclose(torch.sigmoid(model.gate.detach()), torch.tensor(0.5))
    model.zero_grad(set_to_none=True)
    model(ir, vi, fused).sum().backward()
    assert model.chroma.weight.grad.abs().sum() > 0
    assert float(model.chroma.weight.detach().abs().sum()) == 0.0


def test_a_corner_pixel_reaches_the_score():
    torch.manual_seed(3)
    model = BaselineScorer(pretrained=False, global_size=16, tile_size=16).eval()
    ir = torch.rand(1, 1, 30, 28)
    vi = torch.rand(1, 3, 30, 28)
    fused = torch.rand(1, 3, 30, 28, requires_grad=True)
    model(ir, vi, fused).sum().backward()
    assert fused.grad[..., 0, 0].abs().sum() > 0
    assert fused.grad[..., -1, -1].abs().sum() > 0

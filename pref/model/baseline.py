"""Baseline scorer: YCbCr color and a complete tile cover.

Infrared is one grayscale channel. Visible and fusion are Y, Cb, Cr. Y is the
grayscale image. Cb and Cr enter through a zero-initialized stem convolution,
so a neutral color image matches the luma path at step 0. Every tile is
encoded. Overlapping tiles share area by 1/c(p); that splits coverage, and
does not undo the nonlinear reuse of an overlapped pixel. A sigmoid gate
mixes the global view and the tiles and starts at one half.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


def tile_origins(length: int, tile: int) -> list[int]:
    if length <= 0:
        raise ValueError('image length must be positive')
    if tile <= 0:
        raise ValueError('tile size must be positive')
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile + 1, tile))
    last = length - tile
    if origins[-1] != last:
        origins.append(last)
    return origins


def tile_boxes(height: int, width: int, tile: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for top in tile_origins(height, tile):
        block_h = height if height <= tile else tile
        for left in tile_origins(width, tile):
            block_w = width if width <= tile else tile
            boxes.append((top, left, block_h, block_w))
    return boxes


def covers_image(height: int, width: int, tile: int) -> bool:
    seen = torch.zeros(height, width, dtype=torch.bool)
    for top, left, block_h, block_w in tile_boxes(height, width, tile):
        if top + block_h > height or left + block_w > width:
            return False
        seen[top:top + block_h, left:left + block_w] = True
    return bool(seen.all())


def coverage_weights(height: int, width: int, tile: int) -> torch.Tensor:
    """Area each tile owns. A pixel covered c times contributes 1/c to each tile."""
    boxes = tile_boxes(height, width, tile)
    cover = torch.zeros(height, width)
    for top, left, block_h, block_w in boxes:
        cover[top:top + block_h, left:left + block_w] += 1
    if torch.any(cover <= 0):
        raise ValueError('tile cover left a pixel unread')
    weights = [
        (1.0 / cover[top:top + block_h, left:left + block_w]).sum()
        for top, left, block_h, block_w in boxes
    ]
    out = torch.stack(weights)
    if abs(float(out.sum()) - float(height * width)) > 1e-3:
        raise ValueError('coverage weights do not partition the image')
    return out


class BaselineScorer(nn.Module):
    """S(IR, VI, F) for native, aligned images. The caller does not pick a size."""

    def __init__(self, pretrained: bool = True, global_size: int = 224, tile_size: int = 224):
        super().__init__()
        if global_size <= 0 or tile_size <= 0:
            raise ValueError('global_size and tile_size must be positive')
        self.global_size = int(global_size)
        self.tile_size = int(tile_size)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        old = net.conv1
        self.luma = nn.Conv2d(3, old.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.chroma = nn.Conv2d(4, old.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.luma.weight.copy_(old.weight)
            self.chroma.weight.zero_()
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.gate = nn.Parameter(torch.zeros(()))
        mean = torch.tensor([0.485, 0.456, 0.5, 0.5, 0.406, 0.5, 0.5])
        std = torch.tensor([0.229, 0.224, 0.5, 0.5, 0.225, 0.5, 0.5])
        self.register_buffer('mean', mean.view(1, 7, 1, 1))
        self.register_buffer('std', std.view(1, 7, 1, 1))

    def backbone_parameters(self):
        yield from self.luma.parameters()
        yield from self.bn1.parameters()
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            yield from layer.parameters()

    def head_parameters(self):
        yield from self.chroma.parameters()
        yield from self.head.parameters()
        yield self.gate

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone_parameters():
            parameter.requires_grad = trainable

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 7:
            raise ValueError('encoder input must be (N, 7, H, W)')
        image = (image - self.mean) / self.std
        luma = torch.cat([image[:, 0:2], image[:, 4:5]], dim=1)
        chroma = torch.cat([image[:, 2:4], image[:, 5:7]], dim=1)
        hidden = self.bn1(self.luma(luma) + self.chroma(chroma))
        hidden = self.maxpool(self.relu(hidden))
        hidden = self.layer4(self.layer3(self.layer2(self.layer1(hidden))))
        return hidden.mean(dim=(2, 3))

    def forward(self, ir: torch.Tensor, vi: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        ir, vi, fused = _views(ir, vi, fused)
        scores = [self._one(ir[i], vi[i], fused[i]) for i in range(ir.shape[0])]
        return torch.stack(scores)

    def _one(self, ir: torch.Tensor, vi: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        height, width = ir.shape[-2:]
        boxes = tile_boxes(height, width, self.tile_size)
        tiles = []
        for top, left, block_h, block_w in boxes:
            tiles.append(torch.cat([
                ir[:, top:top + block_h, left:left + block_w],
                vi[:, top:top + block_h, left:left + block_w],
                fused[:, top:top + block_h, left:left + block_w],
            ], dim=0))
        local = self.encode(torch.stack(tiles))
        weights = coverage_weights(height, width, self.tile_size).to(device=local.device, dtype=local.dtype)
        weights = weights / weights.sum()
        pooled = (local * weights[:, None]).sum(dim=0, keepdim=True)
        full = self.encode(_resize(torch.cat([ir, vi, fused], dim=0), self.global_size))
        mix = torch.sigmoid(self.gate)
        return self.head((1.0 - mix) * full + mix * pooled).reshape(())


def _resize(image: torch.Tensor, size: int) -> torch.Tensor:
    if image.shape[-2:] == (size, size):
        return image.unsqueeze(0)
    return F.interpolate(image.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False)


def _views(ir: torch.Tensor, vi: torch.Tensor, fused: torch.Tensor):
    if ir.ndim != 4 or vi.ndim != 4 or fused.ndim != 4:
        raise ValueError('expected batched images (B, C, H, W)')
    if ir.shape[0] != vi.shape[0] or ir.shape[0] != fused.shape[0]:
        raise ValueError('IR, visible, and fusion batch sizes differ')
    if ir.shape[-2:] != vi.shape[-2:] or ir.shape[-2:] != fused.shape[-2:]:
        raise ValueError('IR, visible, and fusion spatial sizes differ')
    if ir.shape[1] == 3:
        ir = ir.mean(dim=1, keepdim=True)
    elif ir.shape[1] != 1:
        raise ValueError('infrared must have 1 or 3 channels')
    if vi.shape[1] != 3 or fused.shape[1] != 3:
        raise ValueError('visible and fusion must be Y, Cb, Cr')
    return ir, vi, fused

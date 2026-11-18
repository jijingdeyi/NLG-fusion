"""Color scorer with an explicit region score and a whole-image score.

Infrared is grayscale. Visible and fusion are Y, Cb, Cr, with Y taken from the
grayscale image. The network keeps the native resolution, so the feature grid
grows with the image. Each cell of the layer-2 map is one region and receives
its own score; those scores are averaged. The last two stages then pool to one
whole-image score. A pair of weights, both starting at one half, adds the two
scores.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from pref.model.baseline import _views


class RegionScorer(nn.Module):
    """S(IR, VI, F) from a region score and a whole-image score."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
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
        self.region_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        self.global_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.mix = nn.Parameter(torch.tensor([0.5, 0.5]))
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
        yield from self.region_head.parameters()
        yield from self.global_head.parameters()
        yield self.mix

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone_parameters():
            parameter.requires_grad = trainable

    def _stem(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.mean) / self.std
        luma = torch.cat([image[:, 0:2], image[:, 4:5]], dim=1)
        chroma = torch.cat([image[:, 2:4], image[:, 5:7]], dim=1)
        hidden = self.bn1(self.luma(luma) + self.chroma(chroma))
        return self.layer2(self.layer1(self.maxpool(self.relu(hidden))))

    def forward(self, ir: torch.Tensor, vi: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        ir, vi, fused = _views(ir, vi, fused)
        scores = [self._one(ir[i], vi[i], fused[i]) for i in range(ir.shape[0])]
        return torch.stack(scores)

    def _one(self, ir: torch.Tensor, vi: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        image = torch.cat([ir, vi, fused], dim=0).unsqueeze(0)
        regions = self._stem(image)
        region_score = self.region_head(regions).mean()
        pooled = self.layer4(self.layer3(regions)).mean(dim=(2, 3))
        global_score = self.global_head(pooled).reshape(())
        return self.mix[0] * region_score + self.mix[1] * global_score

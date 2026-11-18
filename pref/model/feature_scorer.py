"""Color scorer that weighs deep feature channels, not spatial positions.

Infrared is grayscale. Visible and fusion are Y, Cb, Cr. The image keeps its
native size. Layer 4 yields one channel per learned feature. Each channel is
pooled over the whole image, so a feature for leaves or people is not tied to
a fixed coordinate. That channel has its own score, and a separate weight says
how much the channel counts in the decision. The weights are shared by both
fusions. A whole-image score is mixed in with weights that start at one half.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from pref.model.baseline import _views


class FeatureScorer(nn.Module):
    """S(IR, VI, F) from weighted deep-feature scores plus a whole-image score."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        old = net.conv1
        channels = int(net.fc.in_features)
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
        self.channel_gain = nn.Parameter(torch.ones(channels))
        self.channel_bias = nn.Parameter(torch.zeros(channels))
        self.channel_logit = nn.Parameter(torch.zeros(channels))
        self.global_head = nn.Sequential(
            nn.Linear(channels, 256),
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
        yield self.channel_gain
        yield self.channel_bias
        yield self.channel_logit
        yield from self.global_head.parameters()
        yield self.mix

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone_parameters():
            parameter.requires_grad = trainable

    def _features(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.mean) / self.std
        luma = torch.cat([image[:, 0:2], image[:, 4:5]], dim=1)
        chroma = torch.cat([image[:, 2:4], image[:, 5:7]], dim=1)
        hidden = self.bn1(self.luma(luma) + self.chroma(chroma))
        hidden = self.layer2(self.layer1(self.maxpool(self.relu(hidden))))
        return self.layer4(self.layer3(hidden))

    def forward(self, ir: torch.Tensor, vi: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        ir, vi, fused = _views(ir, vi, fused)
        scores = [self._one(ir[i], vi[i], fused[i]) for i in range(ir.shape[0])]
        return torch.stack(scores)

    def _one(self, ir: torch.Tensor, vi: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        image = torch.cat([ir, vi, fused], dim=0).unsqueeze(0)
        pooled = self._features(image).mean(dim=(2, 3)).reshape(-1)
        channel_score = self.channel_gain * pooled + self.channel_bias
        channel_weight = torch.softmax(self.channel_logit, dim=0)
        feature_score = (channel_weight * channel_score).sum()
        global_score = self.global_head(pooled.unsqueeze(0)).reshape(())
        return self.mix[0] * feature_score + self.mix[1] * global_score

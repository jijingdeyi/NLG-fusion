"""Stage-1 ResNet-18 scalar scorer (paper S_theta)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNet18ScalarScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat = backbone.fc.in_features
        self.head = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.img_mean) / self.img_std
        z = self.stem(z)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        return self.head(self.pool(z).flatten(1)).squeeze(-1)

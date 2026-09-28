"""Stage-1 scalar scorer: ImageNet ResNet-18 on Concat(IR, VI, F).

No uncertainty head. Verifies that pairwise preferences can train a meaningful s_F
before Stage-2 A/Tie/B + σ modeling (score.md §4 staged plan).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNet18ScalarScorer(nn.Module):
    """X_F = Concat(I_ir, I_vis, F) → ResNet-18 → GAP → MLP → s_F."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        # Keep first conv (3→64); IR/VI/F is treated as a 3-channel stack.
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
            'img_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'img_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        if freeze_backbone:
            self.set_backbone_trainable(False)

    def backbone_parameters(self):
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from m.parameters()

    def head_parameters(self):
        return self.head.parameters()

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone_parameters():
            p.requires_grad = trainable

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x in [0, 1]; apply ImageNet norm for pretrained stem.
        z = (x - self.img_mean) / self.img_std
        z = self.stem(z)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        return self.pool(z).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) → s: (B,)."""
        return self.head(self.encode(x)).squeeze(-1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

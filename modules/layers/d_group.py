import torch
from torch import Tensor, nn

from modules.layers.conv_group import ConvGroup


class DGroup(nn.Module):
    """Parallel dilation conv decoder (no task-level text modulation)."""

    def __init__(self, in_c: int, out_c: int, dim: int, k_size: int, use_bn: bool):
        super().__init__()
        self.conv_d = nn.ModuleList([
            ConvGroup(nn.Conv2d(in_c, dim, kernel_size=k_size, padding='same', dilation=(i + 1)), use_bn=use_bn)
            for i in range(3)
        ])
        self.conv_s = nn.Sequential(
            nn.Conv2d(3 * dim, out_c, kernel_size=3, padding='same'),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        f_x = [conv(x) for conv in self.conv_d]
        return self.conv_s(torch.cat(f_x, dim=1))


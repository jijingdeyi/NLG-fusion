from torch import nn, Tensor


class ConvGroup(nn.Module):
    def __init__(self, conv: nn.Conv2d, use_bn: bool):
        super().__init__()

        # (Conv2d, BN, GELU)
        dim = conv.out_channels
        self.group = nn.Sequential(
            conv,
            nn.BatchNorm2d(dim) if use_bn else nn.Identity(),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.group(x)
    
class ConvGroup_L(nn.Module):
    def __init__(self, conv: nn.Conv2d, use_ln: bool):
        super().__init__()
        self.conv = conv
        self.use_ln = use_ln
        self.gelu = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.use_ln:
            # 动态获取卷积输出的尺寸
            normalized_shape = x.size()[1:]
            layer_norm = nn.LayerNorm(normalized_shape)
            layer_norm.cuda()
            x = layer_norm(x)
        x = self.gelu(x)
        return x



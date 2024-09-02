import torch
from tabulate import tabulate
from torch import nn, Tensor

from modules.layers.conv_group import ConvGroup

##  parallel dilation conv in figure d 


class DGroup(nn.Module):
    """
    [channels: dim, s] -> DGroup -> [channels: 1, s]
    """

    def __init__(self, in_c: int, out_c: int, dim: int, k_size: int, use_bn: bool):
        super().__init__()

        # conv_d: [dim] -> [1]
        self.conv_d = nn.ModuleList([
            ConvGroup(nn.Conv2d(in_c, dim, kernel_size=k_size, padding='same', dilation=(i + 1)), use_bn=use_bn)
            for i in range(3)
        ])

        self.prompt_guidance = FeatureWiseAffine(in_channels=3*dim, text_embed_dim=512)

        # conv_s: [3] -> [1]
        self.conv_s = nn.Sequential(
            nn.Conv2d(3 * dim, out_c, kernel_size=3, padding='same'),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, text_features: Tensor) -> Tensor:
        f_in = x
        # conv_d
        f_x = [conv(f_in) for conv in self.conv_d]
        # suffix
        f_t = torch.cat(f_x, dim=1)
        f_t = self.prompt_guidance(f_t, text_features)
        f_out = self.conv_s(f_t)
        return f_out

    def __str__(self):
        table = [[n, p.mean(), p.grad.mean()] for n, p in self.named_parameters() if p.grad is not None]
        return tabulate(table, headers=['layer', 'weights', 'grad'], tablefmt='pretty')


## Feature Modulation
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels: int, text_embed_dim: int, use_affine_level: bool = True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.in_channels = in_channels
        self.text_embed_dim = text_embed_dim

        # Define the MLP to process text_embed
        self.MLP = nn.Sequential(
            nn.Linear(self.text_embed_dim, self.in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(self.in_channels * 2, self.in_channels * 2 if self.use_affine_level else self.in_channels)
        )

    def forward(self, x: Tensor, text_embed: Tensor) -> Tensor:
        batch = x.shape[0]

        # Process text_embed through MLP
        modulation_params = self.MLP(text_embed)  # shape (batch, in_channels * 2)

        if self.use_affine_level:
            # Split modulation_params into gamma and beta
            gamma, beta = modulation_params.view(batch, -1, 1, 1).chunk(2, dim=1)
            #print(f"gamma shape: {gamma.shape}, beta shape: {beta.shape}, x shape: {x.shape}")
            # Modulate image features
            x = (1 + gamma) * x + beta
        else:
            x = modulation_params.view(batch, self.in_channels, 1, 1)

        return x
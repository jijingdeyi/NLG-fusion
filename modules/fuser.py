import torch
from torch import nn, Tensor

from modules.layers.d_group import DGroup
from modules.layers.transformer import TransformerBlock
from modules.layers.conv_group import ConvGroup
import cv2
import numpy as np


class Fuser(nn.Module):
    def __init__(self, depth: int, dim: int, use_bn: bool, model_clip):
        super().__init__()
        self.depth = depth
        self.model_clip = model_clip
        # attention layer: [2] -> [1], [2] -> [1]
        self.att_a_conv = ConvGroup(nn.Conv2d(4, 1, kernel_size=3, padding='same', bias=False), use_bn=True)
        self.att_b_conv = ConvGroup(nn.Conv2d(4, 1, kernel_size=3, padding='same', bias=False), use_bn=True)

        self.trans_a = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.trans_b = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.trans_ir = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.trans_vi = TransformerBlock(2, num_heads=2, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')

        # prompt guidance
        self.prompt_guidance_ir = FeatureWiseAffine(in_channels=3, text_embed_dim=512)
        self.prompt_guidance_vi = FeatureWiseAffine(in_channels=3, text_embed_dim=512)

        # dilation fuse
        self.decoder = DGroup(in_c=7, out_c=1, dim=dim, k_size=3, use_bn=use_bn)

    def forward(self, i_in: Tensor, text_embed: Tensor, text_embed_ir: Tensor, text_embed_vi: Tensor, init_f: str = 'max', show_detail: bool = False, save_intermediate: bool = False):
        
        # generate f_0 with initial function
        i_1, i_2 = torch.chunk(i_in, chunks=2, dim=1)
        b = i_1.shape[0]
        text_features = self.get_text_feature(text_embed.expand(b, -1)).to(i_1.dtype)
        text_features_ir = self.get_text_feature(text_embed_ir.expand(b, -1)).to(i_1.dtype)
        text_features_vi = self.get_text_feature(text_embed_vi.expand(b, -1)).to(i_1.dtype)

        i_f = [torch.max(i_1, i_2) if init_f == 'max' else (i_1 + i_2) / 2]
        att_a, att_b = [], []

        # loop in subnetwork
        for _ in range(self.depth):
            i_f_x, att_a_x, att_b_x = self._sub_forward(i_1, i_2, i_f[-1], text_features, text_features_ir, text_features_vi)
            i_f.append(i_f_x), att_a.append(att_a_x), att_b.append(att_b_x)

        
        if save_intermediate:
            self.save_images(i_f, att_a, att_b)

        # return as expected
        return (i_f, att_a, att_b) if show_detail else i_f[-1]
    
    
    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature



    def _sub_forward(self, i_1: Tensor, i_2: Tensor, i_f: Tensor, text_features: Tensor, text_features_ir: Tensor, text_features_vi: Tensor ):
        
        # attention
        res_ir, attn_pool_a = self._attention(self.att_a_conv, self.trans_ir, self.trans_a, i_1, i_f)
        res_vi, attn_pool_b = self._attention(self.att_b_conv, self.trans_vi, self.trans_b, i_2, i_f)

        # focus on attention
        i_1_pool = i_1 * attn_pool_a
        i_2_pool = i_2 * attn_pool_b

        # text guidance
        i_in_ir = torch.cat([i_1_pool, res_ir], dim=1)
        i_in_ir_guide = self.prompt_guidance_ir(i_in_ir, text_features_ir)

        i_in_vi = torch.cat([i_2_pool, res_vi], dim=1)
        i_in_vi_guide = self.prompt_guidance_ir(i_in_vi, text_features_vi)

        # dilation fuse
        i_in = torch.cat([i_in_ir_guide, i_f, i_in_vi_guide], dim=1)
        i_out = self.decoder(i_in, text_features)

        # return fusion result of current recurrence
        return i_out, attn_pool_a, attn_pool_b


    @staticmethod
    def _attention(att_conv, trans_base, trans_detail, i_a, i_b):
        i_in = torch.cat([i_a, i_b], dim=1)
        i_max, _ = torch.max(i_in, dim=1, keepdim=True)
        i_avg = torch.mean(i_in, dim=1, keepdim=True)
        
        i_trans_detail = trans_detail(i_in)
        i_trans_base = trans_base(i_in)
        i_in = torch.cat([i_max, i_avg, i_trans_detail], dim=1)
        
        attn_pool = torch.sigmoid(att_conv(i_in))
        return i_trans_base, attn_pool
    
    def save_images(self, i_f_list, att_a_list, att_b_list):
        
        save_path = '/home/ykx/reconet/result/att'
        start_index = max(0, len(i_f_list) - 3)
        for idx, i_f in enumerate(i_f_list[start_index:], start=start_index):
            cv2.imwrite(f'{save_path}/i_f_{idx}.png', self.tensor_to_image(i_f))
        
        for idx, (att_a, att_b) in enumerate(zip(att_a_list, att_b_list)):
            cv2.imwrite(f'{save_path}/att_a_{idx}.png', self.tensor_to_image(att_a))
            cv2.imwrite(f'{save_path}/att_b_{idx}.png', self.tensor_to_image(att_b))


    @staticmethod
    def tensor_to_image(tensor):

        array = tensor.squeeze().detach().cpu().numpy()
        array = (255 * (array - array.min()) / (array.max() - array.min())).astype(np.uint8)
        return array
    
    
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
            # Modulate image features
            x = (1 + gamma) * x + beta
        else:
            x = modulation_params.view(batch, self.in_channels, 1, 1)

        return x
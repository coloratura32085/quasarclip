# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from vitae_prn.NormalCell import NormalCell

from imageutil.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViTAE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=64, patch_size=8, in_chans=5,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, kernel=1,
                 mlp_hidden_dim=None):
        '''
        @Param kernel: int, control the kernel size in PCM
        @Param mlp_hidden_dim: int, the hidden dimenison of FFN, overwrites mlp ratio, default None
        '''
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            NormalCell(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                       kernel=kernel, class_token=True, group=embed_dim // 4, mlp_hidden_dim=mlp_hidden_dim)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)



    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x



def mae_vitae_base_patch16_enc512d8b(**kwargs):
    model = MaskedAutoencoderViTAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vitae_large_patch16_enc512d8b(**kwargs):
    model = MaskedAutoencoderViTAE(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vitae_base_patch8_enc192d4b(**kwargs):
    model = MaskedAutoencoderViTAE(
        patch_size=8, embed_dim=192, depth=12, num_heads=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vitae_base_patch16_enc = mae_vitae_base_patch16_enc512d8b  # decoder: 512 dim, 8 blocks
mae_vitae_large_patch16_enc = mae_vitae_large_patch16_enc512d8b  # decoder: 512 dim, 8 blocks
mae_vitae_base_patch8_enc = mae_vitae_base_patch8_enc192d4b  # decoder: 192 dim, 4 blocks



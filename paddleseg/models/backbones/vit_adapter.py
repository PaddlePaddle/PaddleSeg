# This file is heavily based on https://github.com/czczup/ViT-Adapter

import math
from functools import partial

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.utils import utils, logger
from paddleseg.cvlibs.param_init import normal_init, trunc_normal_init, constant_init
from paddleseg.models.backbones.transformer_utils import to_2tuple, DropPath
from paddleseg.models.layers.vit_adapter_layers import (SpatialPriorModule,
                                                        InteractionBlock,
                                                        deform_inputs)
from paddleseg.models.layers.ms_deformable_attention import MSDeformAttn

__all__ = ['ViTAdapter', 'ViTAdapter_Tiny']


class PatchEmbed(nn.Layer):
    """2D Image to Patch Embedding."""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2D(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose([0, 2, 1])  # BCHW -> BNC
        x = self.norm(x)
        return x, H, W


class Mlp(nn.Layer):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        x_shape = x.shape
        N, C = x_shape[1], x_shape[2]
        qkv = self.qkv(x).reshape(
            (-1, N, 3, self.num_heads, C // self.num_heads)).transpose(
                (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma1 = self.create_parameter(
                shape=(dim, ),
                default_initializer=paddle.nn.initializer.Constant(value=1.))
            self.gamma2 = self.create_parameter(
                shape=(dim, ),
                default_initializer=paddle.nn.initializer.Constant(value=1.))

    def forward(self, x, H, W):
        if self.layer_scale:
            x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Layer):
    """Vision Transformer.

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 layer_scale=True,
                 embed_layer=PatchEmbed,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
                 act_layer=nn.GELU,
                 pretrained=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_channels (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            pretrained: (str): pretrained path
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        act_layer = act_layer or nn.GELU
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.pretrain_size = img_size
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate

        self.patch_embed = embed_layer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_chans=in_channels,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + self.num_tokens, embed_dim),
            default_initializer=paddle.nn.initializer.Constant(value=0.))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate,
                          depth)  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  act_layer=act_layer,
                  layer_scale=layer_scale) for i in range(depth)
        ])

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        utils.load_pretrained_model(self, self.pretrained)


@manager.BACKBONES.add_component
class ViTAdapter(VisionTransformer):
    """ The ViT-Adapter
    """

    def __init__(self,
                 pretrain_size=224,
                 num_heads=12,
                 conv_inplane=64,
                 n_points=4,
                 deform_num_heads=6,
                 init_values=0.,
                 interaction_indexes=None,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 deform_ratio=1.0,
                 add_vit_feature=True,
                 pretrained=None,
                 use_extra_extractor=True,
                 *args,
                 **kwargs):

        super().__init__(num_heads=num_heads,
                         pretrained=pretrained,
                         *args,
                         **kwargs)

        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim
        self.feat_channels = [embed_dim] * 4

        self.level_embed = self.create_parameter(
            shape=(3, embed_dim),
            default_initializer=paddle.nn.initializer.Constant(value=0.))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=embed_dim)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim,
                             num_heads=deform_num_heads,
                             n_points=n_points,
                             init_values=init_values,
                             drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer,
                             with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio,
                             deform_ratio=deform_ratio,
                             extra_extractor=(
                                 (True if i == len(interaction_indexes) -
                                  1 else False) and use_extra_extractor))
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.Conv2DTranspose(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_init(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                constant_init(m.bias, value=0)
        elif isinstance(m, nn.LayerNorm) or isinstance(
                m, (nn.BatchNorm2D, nn.SyncBatchNorm)):
            constant_init(m.bias, value=0)
            constant_init(m.weight, value=1.0)
        elif isinstance(m, nn.Conv2D) or isinstance(m, nn.Conv2DTranspose):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            normal_init(m.weight, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                constant_init(m.bias, value=0)

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            [1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16,
             -1]).transpose([0, 3, 1, 2])
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape([1, -1, H * W]).transpose([0, 2, 1])
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = paddle.concat([c2, c3, c4], axis=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose([0, 2, 1]).reshape([bs, dim, H, W]))

        # Split & Reshape
        c2 = c[:, 0:c2.shape[1], :]
        c3 = c[:, c2.shape[1]:c2.shape[1] + c3.shape[1], :]
        c4 = c[:, c2.shape[1] + c3.shape[1]:, :]

        c2 = c2.transpose([0, 2, 1]).reshape([bs, dim, H * 2, W * 2])
        c3 = c3.transpose([0, 2, 1]).reshape([bs, dim, H, W])
        c4 = c4.transpose([0, 2, 1]).reshape([bs, dim, H // 2, W // 2])
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1,
                               scale_factor=4,
                               mode='bilinear',
                               align_corners=False)
            x2 = F.interpolate(x2,
                               scale_factor=2,
                               mode='bilinear',
                               align_corners=False)
            x4 = F.interpolate(x4,
                               scale_factor=0.5,
                               mode='bilinear',
                               align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


@manager.BACKBONES.add_component
def ViTAdapter_Tiny(**kwargs):
    return ViTAdapter(num_heads=3,
                      patch_size=16,
                      embed_dim=192,
                      depth=12,
                      mlp_ratio=4,
                      drop_path_rate=0.1,
                      conv_inplane=64,
                      n_points=4,
                      deform_num_heads=6,
                      cffn_ratio=0.25,
                      deform_ratio=1.0,
                      interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
                      **kwargs)

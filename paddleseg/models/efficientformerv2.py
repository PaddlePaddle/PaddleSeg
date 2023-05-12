# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import itertools
import copy
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager, param_init
from paddleseg.utils import utils, logger
from paddleseg.models.backbones.transformer_utils import to_2tuple, zeros_, DropPath


@manager.MODELS.add_component
class EfficientFormerV2(nn.Layer):
    """
    The EfficientFormerV2 implementation based on PaddlePaddle.

    The original article refers to "Yanyu Li, Ju Hu, Yang Wen, Georgios Evangelidis,
    Kamyar Salahi, Yanzhi Wang, Sergey Tulyakov, Jian Ren. Rethinking Vision
    Transformers for MobileNet Size and Speed. arXiv preprint arXiv:2212.08059 (2022)."

    Args:
         layers (List): The depth of every stage.
         embed_dims (List): The width of every stage.
         mlp_ratios (int, optional): The radio of the mlp. Default:4
         downsamples (list[bool]): whether use the downsamples in the model.
         pool_size (int, optional): The kernel size of the pool layer. Default:3
         norm_layer (layer, optional): The norm layer type. Default: nn.BatchNorm2D
         act_layer (layer, optional): The activate function type. Default: nn.GELU
         num_classes (int, optional): The num of the classes. Default: 150
         down_patch_size (int, optional): The patch size of down sample. Default: 3
         down_stride (int, optional): The stride size of the down sample. Default: 2
         down_pad (int, optional): The padding size of the down sample. Default: 1
         drop_rate (float, optional): The drop rate in meta_blocks. Default: 0.
         drop_path_rate (float): The drop path rate in meta_blocks. Default: 0.02
         use_layer_scale (bool, optional): Whether use the multi-scale layer. Default: True
         layer_scale_init_value (float, optional): The initial value of token. Default: 1e-5
         fork_feat (bool): Whether use the classes as the the output channels .
         pretrained (str): The path or url of pretrained model.
         vit_num (str): The num of vit stages.
         distillation (bool, optional): Whether use the distillation. Default:True
         resolution (int, optional): The resolution of the input image. Default
         e_ratios (int): The ratios in the meta_blocks.
  }
    """
    def __init__(self,
                 layers,
                 embed_dims=None,
                 mlp_ratios=4,
                 downsamples=None,
                 pool_size=3,
                 norm_layer=nn.BatchNorm2D,
                 act_layer=nn.GELU,
                 num_classes=150,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5,
                 fork_feat=False,
                 pretrained=None,
                 vit_num=0,
                 distillation=True,
                 resolution=512,
                 e_ratios=None):
        super().__init__()

        self.pretrained = pretrained

        self.feat_channels = embed_dims

        self.fpn_head = FPN_head(num_classes,
                                embed_dims,
                                drop_path_rate)

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = stem(3, embed_dims[0], act_layer=act_layer)

        network = []
        for i in range(len(layers)):
            stage = meta_blocks(embed_dims[i], i, layers,
                                pool_size=pool_size, mlp_ratio=mlp_ratios,
                                act_layer=act_layer, norm_layer=norm_layer,
                                drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate,
                                use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                resolution=math.ceil(resolution / (2 ** (i + 2))),
                                vit_num=vit_num,
                                e_ratios=e_ratios)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                if i >= 2:
                    asub = True
                else:
                    asub = False
                network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1],
                        resolution=math.ceil(resolution / (2 ** (i + 2))),
                        asub=asub,
                        act_layer=act_layer, norm_layer=norm_layer,
                    )
                )

        self.network = nn.LayerList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(
                    embed_dims[-1], num_classes) if num_classes > 0 \
                    else nn.Identity()

        self.init_weight()


    # imagenet pre-trained weights
    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
            elif isinstance(layer, nn.Linear):
                param_init.normal_init(layer.weight)
                if layer.bias is not None:
                    param_init.constant_init(layer.bias, value=0)
            elif isinstance(layer, nn.LayerNorm):
                param_init.constant_init(layer.bias, value=0)
                param_init.constant_init(layer.weight, value=1)

        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)


    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                outs.append(x)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        H, W = x.shape[-2:]
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        x = self.fpn_head(x)
        out = [F.interpolate(x, size=[H, W], mode='bilinear', align_corners=False)]
        return out

class Attention4D(nn.Layer):
    def __init__(self,
                 dim=384,
                 key_dim=32,
                 num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 act_layer=nn.ReLU,
                 stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(nn.Conv2D(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                                             nn.BatchNorm2D(dim), )
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        self.N = self.resolution ** 2
        self.N2 = self.N
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.q = nn.Sequential(nn.Conv2D(dim, self.num_heads * self.key_dim, 1),
                               nn.BatchNorm2D(self.num_heads * self.key_dim), )
        self.k = nn.Sequential(nn.Conv2D(dim, self.num_heads * self.key_dim, 1),
                               nn.BatchNorm2D(self.num_heads * self.key_dim), )
        self.v = nn.Sequential(nn.Conv2D(dim, self.num_heads * self.d, 1),
                               nn.BatchNorm2D(self.num_heads * self.d),
                               )
        self.v_local = nn.Sequential(nn.Conv2D(self.num_heads * self.d, self.num_heads * self.d,
                                               kernel_size=3, stride=1, padding=1, groups=self.num_heads * self.d),
                                     nn.BatchNorm2D(self.num_heads * self.d), )
        self.talking_head1 = nn.Conv2D(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        self.talking_head2 = nn.Conv2D(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)

        self.proj = nn.Sequential(act_layer(),
                                  nn.Conv2D(self.dh, dim, 1),
                                  nn.BatchNorm2D(dim), )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.register_buffer('attention_biases', paddle.zeros([num_heads, 49]))
        self.register_buffer('attention_bias_idxs',
                             paddle.ones([49, 49], dtype=paddle.int64))

        self.attention_biases_seg = self.create_parameter(shape=[num_heads, len(attention_offsets)],
                                                          default_initializer=zeros_)
        self.register_buffer('attention_bias_idxs_seg',
                             paddle.to_tensor(idxs).reshape([N, N]))


    @paddle.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = paddle.gather(self.attention_biases_seg,
                             self.attention_bias_idxs_seg.flatten(),
                             axis=1).reshape([self.attention_biases_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[1]])


    def forward(self, x):
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)
            H = H // 2
            W = W // 2

        q = self.q(x).flatten(2).reshape([B, self.num_heads, -1, H * W]).transpose([0, 1, 3, 2])
        k = self.k(x).flatten(2).reshape([B, self.num_heads, -1, H * W]).transpose([0, 1, 2, 3])
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape([B, self.num_heads, -1, H * W]).transpose([0, 1, 3, 2])

        attn = (q @ k) * self.scale
        bias = paddle.gather(self.attention_biases_seg,
                             self.attention_bias_idxs_seg.flatten(),
                             axis=1).reshape([self.attention_biases_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[1]])

        bias = F.interpolate(bias.unsqueeze(0), size=attn.shape[-2:], mode='bicubic')
        attn = attn + bias

        attn = self.talking_head1(attn)
        attn = F.softmax(attn, axis=-1)
        attn = self.talking_head2(attn)

        x = (attn @ v)

        out = x.transpose([0, 1, 3, 2]).reshape([B, self.dh, H, W]).clone() + v_local
        if self.upsample is not None:
            out = self.upsample(out)

        out = self.proj(out)
        return out


def stem(in_chs, out_chs, act_layer=nn.ReLU):
    return nn.Sequential(
        nn.Conv2D(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2D(out_chs // 2),
        act_layer(),
        nn.Conv2D(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2D(out_chs),
        act_layer(),
    )


class LGQuery(nn.Layer):
    def __init__(self, in_dim, out_dim, resolution1, resolution2):
        super().__init__()
        self.resolution1 = resolution1
        self.resolution2 = resolution2
        self.pool = nn.AvgPool2D(1, 2, 0)
        self.local = nn.Sequential(nn.Conv2D(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim),
                                   )
        self.proj = nn.Sequential(nn.Conv2D(in_dim, out_dim, 1),
                                  nn.BatchNorm2D(out_dim), )

    def forward(self, x):
        B, C, H, W = x.shape
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)
        return q


class Attention4DDownsample(nn.Layer):
    def __init__(self, dim=384, key_dim=16, num_heads=8,
                 attn_ratio=4,
                 resolution=7,
                 out_dim=None,
                 act_layer=None,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        self.resolution = resolution

        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        if out_dim is not None:
            self.out_dim = out_dim
        else:
            self.out_dim = dim

        self.resolution2 = math.ceil(self.resolution / 2)
        self.q = LGQuery(dim, self.num_heads * self.key_dim, self.resolution, self.resolution2)

        self.N = self.resolution ** 2
        self.N2 = self.resolution2 ** 2

        self.k = nn.Sequential(nn.Conv2D(dim, self.num_heads * self.key_dim, 1),
                               nn.BatchNorm2D(self.num_heads * self.key_dim), )
        self.v = nn.Sequential(nn.Conv2D(dim, self.num_heads * self.d, 1),
                               nn.BatchNorm2D(self.num_heads * self.d),
                               )
        self.v_local = nn.Sequential(nn.Conv2D(self.num_heads * self.d, self.num_heads * self.d,
                                               kernel_size=3, stride=2, padding=1, groups=self.num_heads * self.d),
                                     nn.BatchNorm2D(self.num_heads * self.d), )

        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2D(self.dh, self.out_dim, 1),
            nn.BatchNorm2D(self.out_dim), )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        points_ = list(itertools.product(
            range(self.resolution2), range(self.resolution2)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * math.ceil(self.resolution / self.resolution2) - p2[0] + (size - 1) / 2),
                    abs(p1[1] * math.ceil(self.resolution / self.resolution2) - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.register_buffer('attention_biases', paddle.zeros([num_heads, 196]))
        self.register_buffer('attention_bias_idxs',
                             paddle.ones([49, 196], dtype=paddle.int64))

        self.attention_biases_seg = self.create_parameter(shape=[num_heads, len(attention_offsets)],
                                                          default_initializer=zeros_ )
        self.register_buffer('attention_bias_idxs_seg',
                             paddle.to_tensor(idxs, dtype=paddle.int64).reshape([N_, N]))


    @paddle.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = paddle.gather(self.attention_biases_seg,
                             self.attention_bias_idxs_seg.flatten(),
                             axis=1).reshape([self.attention_biases_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[1]])

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.q(x).flatten(2).reshape([B, self.num_heads, -1, H * W // 4]).clone().transpose([0, 1, 3, 2])
        k = self.k(x).flatten(2).reshape([B, self.num_heads, -1, H * W]).clone().transpose([0, 1, 2, 3])
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape([B, self.num_heads, -1, H * W]).clone().transpose([0, 1, 3, 2])

        attn = (q @ k) * self.scale
        bias = paddle.gather(self.attention_biases_seg,
                             self.attention_bias_idxs_seg.flatten(),
                             axis=1).reshape([self.attention_biases_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[1]])
        bias = F.interpolate(bias.unsqueeze(0), size=attn.shape[-2:], mode='bicubic')
        attn = attn + bias

        attn = F.softmax(attn, axis=-1)

        x = (attn @ v).transpose([0, 1, 3, 2])
        out = x.reshape([B, self.dh, H // 2, W // 2]) + v_local

        out = self.proj(out)
        return out


class Embedding(nn.Layer):
    def __init__(self,
                 patch_size=3,
                 stride=2,
                 padding=1,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=nn.BatchNorm2D,
                 light=False,
                 asub=False,
                 resolution=None,
                 act_layer=nn.ReLU,
                 attn_block=Attention4DDownsample):
        super().__init__()
        self.light = light
        self.asub = asub

        if self.light:
            self.new_proj = nn.Sequential(
                nn.Conv2D(in_chans, in_chans, kernel_size=3, stride=2, padding=1, groups=in_chans),
                nn.BatchNorm2D(in_chans),
                nn.Hardswish(),
                nn.Conv2D(in_chans, embed_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(embed_dim),
            )
            self.skip = nn.Sequential(
                nn.Conv2D(in_chans, embed_dim, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2D(embed_dim)
            )
        elif self.asub:
            self.attn = attn_block(dim=in_chans, out_dim=embed_dim,
                                   resolution=resolution, act_layer=act_layer)

            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.conv = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
            self.bn = norm_layer(embed_dim) if norm_layer else nn.Identity()
        else:
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size,
                                  stride=stride, padding=padding)
            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.light:
            out = self.new_proj(x) + self.skip(x)
        elif self.asub:
            out_conv = self.conv(x)
            out_conv = self.bn(out_conv)

            out = self.attn(x) + out_conv

        else:
            x = self.proj(x)
            out = self.norm(x)
        return out


class Mlp(nn.Layer):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.init_weight()

        if self.mid_conv:
            self.mid = nn.Conv2D(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features)
            #self.mid_norm = nn.BatchNorm2D(hidden_features)

        self.norm1 = nn.BatchNorm2D(hidden_features)
        self.norm2 = nn.BatchNorm2D(out_features)
        self.init_weight()

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight)
                if layer.bias is not None:
                    zeros_(layer.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            #x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = self.drop(x)
        return x


class AttnFFN(nn.Layer):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm,
                 drop=0.,
                 drop_path=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5,
                 resolution=7,
                 stride=None):

        super().__init__()

        self.token_mixer = Attention4D(dim, resolution=resolution, act_layer=act_layer, stride=stride)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            param_value = layer_scale_init_value * paddle.ones([dim]).unsqueeze(-1).unsqueeze(-1)
            self.layer_scale_1 = self.create_parameter(shape=param_value.shape,
                                                       dtype=str(param_value.numpy().dtype),
                                                       default_initializer=nn.initializer.Assign(param_value))
            self.layer_scale_2 = self.create_parameter(shape=param_value.shape,
                                                       dtype=str(param_value.numpy().dtype),
                                                       default_initializer=nn.initializer.Assign(param_value))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))

        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


class FFN(nn.Layer):
    def __init__(self,
                 dim,
                 pool_size=3,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0.,
                 drop_path=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            param_value_2 = layer_scale_init_value * paddle.ones([dim]).unsqueeze(-1).unsqueeze(-1)
            self.layer_scale_2 = self.create_parameter(shape=param_value_2.shape,
                                                       dtype=str(param_value_2.numpy().dtype),
                                                       default_initializer=nn.initializer.Assign(param_value_2))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


def meta_blocks(dim, index, layers,
                pool_size=3, mlp_ratio=4.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                drop_rate=.0, drop_path_rate=0.,
                use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1, resolution=7, e_ratios=None):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        mlp_ratio = e_ratios[str(index)][block_idx]
        if index >= 2 and block_idx > layers[index] - 1 - vit_num:
            if index == 2:
                stride = 2
            else:
                stride = None
            blocks.append(AttnFFN(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                resolution=resolution,
                stride=stride,
            ))
        else:
            blocks.append(FFN(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))

    blocks = nn.Sequential(*blocks)
    return blocks


class FPN_head(nn.Layer):
    def __init__(self,
                 num_classes,
                 embed_dims,
                 drop_path_tate,
                 align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.neck = FPN(in_channels = embed_dims,
                        out_channels= 256,
                        num_outs = 4)
        self.head = FPNHead(in_channels=[256, 256, 256, 256],
                            in_index=[0, 1, 2, 3],
                            feature_strides=[4, 8, 16, 32],
                            channels=128,
                            dropout_ratio=drop_path_tate,
                            num_classes=num_classes)

    def forward(self, x):
        x = self.neck(x)
        x = self.head(x)

        return x


class FPN(nn.Layer):
    """Feature Pyramid Network.

    This neck is the implementation of `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=False,
                 relu_before_extra_convs=False):
        super(FPN, self).__init__()

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.LayerList()
        self.fpn_convs = nn.LayerList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2D(in_channels[i], out_channels, 1)
            fpn_conv = nn.Conv2D(out_channels, out_channels, 3, padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = nn.Conv2D(in_channels,
                                           out_channels,
                                           3,
                                           stride=2,
                                           padding=1)
                self.fpn_convs.append(extra_fpn_conv)
        self.init_weight()

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.

            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, mode='nearest', align_corners=False)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

    def init_weight(self):
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                param_init.normal_init(sublayer.weight, std=0.001)
            elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(sublayer.weight, value=1.0)
                param_init.constant_init(sublayer.bias, value=0.0)


class Upsample(nn.Layer):
    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):

        super().__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return F.interpolate(x, size, None, self.mode, self.align_corners)


class FPNHead(nn.Layer):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """
    def __init__(self,
                 in_index = [0, 1, 2, 3],
                 in_channels=[256, 256, 256, 256],
                 channels = 128,
                 feature_strides = [4, 8, 16, 32],
                 dropout_ratio = 0.1,
                 num_classes = 150,
                 align_corners=False):
        super().__init__()

        self.in_channels = in_channels

        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]

        self.in_index = in_index

        self.channels = channels
        self.feature_strides = feature_strides
        self.align_corners = align_corners
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2D(self.dropout_ratio)

        self.scale_heads = nn.LayerList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    nn.Sequential(nn.Conv2D(self.in_channels[i] if k == 0 else self.channels,
                                            self.channels,
                                            3,
                                            padding=1,
                                            bias_attr=False),
                                  nn.BatchNorm2D(self.channels),
                                  nn.ReLU()))


                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))

            self.scale_heads.append(nn.Sequential(*scale_head))
        self.cls_seg = nn.Conv2D(self.channels, self.num_classes, kernel_size=1)
        self.init_weight()

    def forward(self, inputs):

        x = [inputs[i] for i in self.in_index]

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + F.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        if self.dropout_ratio > 0:
            output = self.dropout(output)
        output = self.cls_seg(output)
        return output

    def init_weight(self):
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                param_init.normal_init(sublayer.weight, std=0.001)
            elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(sublayer.weight, value=1.0)
                param_init.constant_init(sublayer.bias, value=0.0)

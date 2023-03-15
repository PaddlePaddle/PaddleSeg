import math
import itertools
import copy

import paddle
import paddle.nn as nn
import numpy as np

from paddleseg.cvlibs import manager, param_init
from paddleseg.utils import utils, logger
import paddle.nn.functional as F

from paddleseg.models.backbones.transformer_utils import *
from paddleseg.models.pfpnnet import PFPNNet


__all__ = [
    'EfficientFormerv2_s0', 'EfficientFormerv2_s1',
    'EfficientFormerv2_s2', 'EfficientFormerv2_l'
]

EfficientFormer_width = {
    'L': [40, 80, 192, 384],  # 26m 83.3% 6attn
    'S2': [32, 64, 144, 288],  # 12m 81.6% 4attn dp0.02
    'S1': [32, 48, 120, 224],  # 6.1m 79.0
    'S0': [32, 48, 96, 176],  # 75.0 75.7
}

EfficientFormer_depth = {
    'L': [5, 5, 15, 10],  # 26m 83.3%
    'S2': [4, 4, 12, 8],  # 12m
    'S1': [3, 3, 9, 6],  # 79.0
    'S0': [2, 2, 6, 4],  # 75.7
}

# 26m
expansion_ratios_L = {
    '0': [4, 4, 4, 4, 4],
    '1': [4, 4, 4, 4, 4],
    '2': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
}

# 12m
expansion_ratios_S2 = {
    '0': [4, 4, 4, 4],
    '1': [4, 4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 3, 3, 3, 3, 4, 4],
}

# 6.1m
expansion_ratios_S1 = {
    '0': [4, 4, 4],
    '1': [4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 4, 4, 4],
    '3': [4, 4, 3, 3, 4, 4],
}

# 3.5m
expansion_ratios_S0 = {
    '0': [4, 4],
    '1': [4, 4],
    '2': [4, 3, 3, 3, 4, 4],
    '3': [4, 3, 3, 4],
}


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
        # self.register_buffer('attention_bias_idxs',
        #                      torch.LongTensor(idxs).view(N, N))

    @paddle.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            #self.ab = self.attention_biases_seg[:, self.attention_bias_idxs_seg]
            self.ab = paddle.gather(self.attention_biases_seg,
                             self.attention_bias_idxs_seg.flatten(),
                             axis=1).reshape([self.attention_biases_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[1]])

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)
            H = H // 2
            W = W // 2

        q = self.q(x).flatten(2).reshape([B, self.num_heads, -1, H * W]).transpose([0, 1, 3, 2])
        #q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape([B, self.num_heads, -1, H * W]).transpose([0, 1, 2, 3])
        #k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape([B, self.num_heads, -1, H * W]).transpose([0, 1, 3, 2])
        #v = v.flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 3, 2)

        attn = (q @ k) * self.scale
        #bias = self.attention_biases_seg[:, self.attention_bias_idxs_seg] if self.training else self.ab
        bias = paddle.gather(self.attention_biases_seg,
                             self.attention_bias_idxs_seg.flatten(),
                             axis=1).reshape([self.attention_biases_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[1]]) #if self.training else self.ab

        bias = F.interpolate(bias.unsqueeze(0), size=attn.shape[-2:], mode='bicubic')
        attn = attn + bias

        attn = self.talking_head1(attn)
        attn = F.softmax(attn, axis=-1)
        attn = self.talking_head2(attn)

        x = (attn @ v)

        out = x.transpose([0, 1, 3, 2]).reshape([B, self.dh, H, W]).clone() + v_local
        #out = x.transpose(2, 3).reshape(B, self.dh, H, W) + v_local
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
        # self.attention_biases = torch.nn.Parameter(
        #     torch.zeros(num_heads, len(attention_offsets)))
        # self.register_buffer('attention_bias_idxs',
        #                      torch.LongTensor(idxs).view(N_, N))

    @paddle.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            #self.ab = self.attention_biases_seg[:, self.attention_bias_idxs_seg]
            self.ab = paddle.gather(self.attention_biases_seg,
                             self.attention_bias_idxs_seg.flatten(),
                             axis=1).reshape([self.attention_biases_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[1]])

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape

        q = self.q(x).flatten(2).reshape([B, self.num_heads, -1, H * W // 4]).clone().transpose([0, 1, 3, 2])
        #q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, H * W // 4).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape([B, self.num_heads, -1, H * W]).clone().transpose([0, 1, 2, 3])
        #k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape([B, self.num_heads, -1, H * W]).clone().transpose([0, 1, 3, 2])
        #v = v.flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 3, 2)


        attn = (q @ k) * self.scale
        #bias = self.attention_biases_seg[:, self.attention_bias_idxs_seg] if self.training else self.ab
        bias = paddle.gather(self.attention_biases_seg,
                             self.attention_bias_idxs_seg.flatten(),
                             axis=1).reshape([self.attention_biases_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[0],
                                              self.attention_bias_idxs_seg.shape[1]]) #if self.training else self.ab
        bias = F.interpolate(bias.unsqueeze(0), size=attn.shape[-2:], mode='bicubic')
        attn = attn + bias

        attn = F.softmax(attn, axis=-1)

        x = (attn @ v).transpose([0, 1, 3, 2])
        out = x.reshape([B, self.dh, H // 2, W // 2]) + v_local

        out = self.proj(out)
        return out


"""Copy from timm"""
from itertools import repeat
import collections.abc
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


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
            self.mid_norm = nn.BatchNorm2D(hidden_features)

        self.norm1 = nn.BatchNorm2D(hidden_features)
        # self.norm1 = PruneBatchNorm2D(hidden_features)
        self.norm2 = nn.BatchNorm2D(out_features)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Conv2D):
    #         param_init.normal_init(m.weight, std=.02)
    #         if m.bias is not None:
    #             param_init.constant_init(m.bias, value=0)
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
            x_mid = self.mid_norm(x_mid)
            # x = self.act(x_mid + x)
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
            # self.layer_scale_1 = nn.Parameter(
            #     layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
            # self.layer_scale_2 = nn.Parameter(
            #     layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
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
            # self.layer_scale_2 = nn.Parameter(
            #     layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

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


# 构建Conv+BN+ReLU块：
class ConvBnReLU(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


# 构建FPN整体，参数base_channels为网络设置的基础通道数，每降低一个分辨率，通道数依据此翻倍
# forward输入：三通道图像
# forward输出：四个尺度的特征
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

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 #extra_convs_on_inputs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 #conv_cfg=None,
                 #norm_cfg=None,
                 #act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 #init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

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
        # assert isinstance(add_extra_convs, (str, bool))
        # if isinstance(add_extra_convs, str):
        #     # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
        #     assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        # elif add_extra_convs:  # True
        #     if extra_convs_on_inputs:
        #         # For compatibility with previous release
        #         # TODO: deprecate `extra_convs_on_inputs`
        #         self.add_extra_convs = 'on_input'
        #     else:
        #         self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.LayerList()
        self.fpn_convs = nn.LayerList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvBnReLU(in_channels[i], channels, 1, 1, 0)
            # l_conv = ConvModule(
            #     in_channels[i],
            #     out_channels,
            #     1,
            #     conv_cfg=conv_cfg,
            #     norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
            #     act_cfg=act_cfg,
            #     inplace=False)
            fpn_conv = ConvBnReLU(channels, channels, 3, 1, 1)
            # fpn_conv = ConvModule(
            #     out_channels,
            #     out_channels,
            #     3,
            #     padding=1,
            #     conv_cfg=conv_cfg,
            #     norm_cfg=norm_cfg,
            #     act_cfg=act_cfg,
            #     inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = channels
                extra_fpn_conv = ConvBnReLU(in_channels, channels, 3, 2, 1)
                # extra_fpn_conv = ConvModule(
                #     in_channels,
                #     out_channels,
                #     3,
                #     stride=2,
                #     padding=1,
                #     conv_cfg=conv_cfg,
                #     norm_cfg=norm_cfg,
                #     act_cfg=act_cfg,
                #     inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

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
            if 'scale_factor' in self.upsample_cfg:
                # laterals[i - 1] += resize(laterals[i], **self.upsample_cfg)
                laterals[i - 1] += F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                # laterals[i - 1] += resize(
                #     laterals[i], size=prev_shape, **self.upsample_cfg)
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

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
        return outs


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
                 in_channels,
                 channels,
                 numclasses,
                 feature_strides,
                 ):
        super(FPNHead, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = numclasses

        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.LayerList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(ConvBnReLU(
                    self.in_channels[i] if k == 0 else self.channels,
                    self.channels,
                    3,
                    1,
                    1))
                    # ConvModule(
                    #     self.in_channels[i] if k == 0 else self.channels,
                    #     self.channels,
                    #     3,
                    #     padding=1,
                    #     conv_cfg=self.conv_cfg,
                    #     norm_cfg=self.norm_cfg,
                    #     act_cfg=self.act_cfg)
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        nn.Upsample(scale_factor=2,
                                      mode='bilinear',
                                      align_corners=False))
                        # Upsample(
                        #     scale_factor=2,
                        #     mode='bilinear',
                        #     align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))
        self.predict = nn.Sequential(nn.Upsample(scale_factor=4,
                                                 mode='bilinear',
                                                 align_corners=False),
                                     nn.Conv2D(self.channels, self.num_classes, kernel_size=1))

    def forward(self, inputs):

        # upsampled_inputs = [F.interpolate(
        #             x,
        #             size=inputs[0].shape[2:],
        #             mode='bilinear',
        #             align_corners=False) for x in inputs]
        # x = paddle.concat(upsampled_inputs, axis=1)
        #x = self._transform_inputs(inputs)
        x = inputs

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + F.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=False)

        #output = self.cls_seg(output)
        output = self.predict(output)
        return output




class EfficientFormerV2(nn.Layer):

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
                 e_ratios=expansion_ratios_S2,
                 fpn_in_channels=None,
                 channels=None,
                 feature_strides=None,
                 **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.feat_channels = embed_dims

        # self.neck = FPN(fpn_in_channels,
        #                 channels,
        #                 len(fpn_in_channels))

        self.fpnhead = FPNHead(fpn_in_channels,
                               channels,
                               num_classes,
                               feature_strides
                               )

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

        # self.init_cfg = copy.deepcopy(init_cfg)
        # # load pre-trained model
        # if self.fork_feat and (
        #         self.init_cfg is not None or pretrained is not None):
        #     self.init_weights()
        #     #self = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(self)
        #     paddle.nn.SyncBatchNorm.convert_sync_batchnorm(self)
        #     self.train()

    # # init for classification
    # def cls_init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_init(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             constant_init(m.bias, value=0)

    # init for mmdetection or mmsegmentation by loading
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

    # @paddle.no_grad()
    # def train(self, mode=True):
    #     super().train(mode)
    #     for m in self.modules():
    #         # trick: eval have effect on BatchNorm only
    #         if isinstance(m, nn.BatchNorm2D):
    #             m.eval()

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                # norm_layer = getattr(self, f'norm{idx}')
                # x_out = norm_layer(x)
                outs.append(x)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        out = self.fpnhead(x)   # N,num_classes,H,W

        return [out]


@manager.MODELS.add_component
class EfficientFormerv2_s0(EfficientFormerV2):
    def __init__(self, **kwargs):
        super().__init__(
            layers=EfficientFormer_depth['S0'],
            embed_dims=EfficientFormer_width['S0'],
            downsamples=[True, True, True, True],
            fork_feat=True,
            drop_path_rate=0.,
            vit_num=2,
            e_ratios=expansion_ratios_S0,
            **kwargs)


@manager.MODELS.add_component
class EfficientFormerv2_s1(EfficientFormerV2):
    def __init__(self, **kwargs):
        super().__init__(
            layers=EfficientFormer_depth['S1'],
            embed_dims=EfficientFormer_width['S1'],
            downsamples=[True, True, True, True],
            fork_feat=True,
            drop_path_rate=0.,
            vit_num=2,
            e_ratios=expansion_ratios_S1,
            **kwargs)


@manager.MODELS.add_component
class EfficientFormerv2_s2(EfficientFormerV2):
    def __init__(self, **kwargs):
        super().__init__(
            layers=EfficientFormer_depth['S2'],
            embed_dims=EfficientFormer_width['S2'],
            downsamples=[True, True, True, True],
            fork_feat=True,
            drop_path_rate=0.02,
            vit_num=4,
            e_ratios=expansion_ratios_S2,
            **kwargs)



@manager.MODELS.add_component
class EfficientFormerv2_l(EfficientFormerV2):
    def __init__(self, **kwargs):
        super().__init__(
            layers=EfficientFormer_depth['L'],
            embed_dims=EfficientFormer_width['L'],
            downsamples=[True, True, True, True],
            fork_feat=True,
            drop_path_rate=0.1,
            vit_num=6,
            e_ratios=expansion_ratios_L,
            **kwargs)

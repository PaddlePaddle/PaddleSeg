# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/MIC-DKFZ/nnUNet

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import numpy as np
from itertools import repeat

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from medicalseg.cvlibs import manager
from medicalseg.utils import utils


class PatchMerging(nn.Layer):

    def __init__(self, dim, norm_layer=nn.LayerNorm, tag=None):
        super().__init__()
        self.dim = dim
        if tag == 0:
            self.reduction = nn.Conv3D(dim,
                                       dim * 2,
                                       kernel_size=[1, 3, 3],
                                       stride=[1, 2, 2],
                                       padding=[0, 1, 1])
        elif tag == 1:
            self.reduction = nn.Conv3D(dim,
                                       dim * 2,
                                       kernel_size=[3, 3, 3],
                                       stride=[2, 2, 2],
                                       padding=[1, 1, 1])
        else:
            self.reduction = nn.Conv3D(dim,
                                       dim * 2,
                                       kernel_size=[3, 3, 3],
                                       stride=[2, 2, 2],
                                       padding=[0, 1, 1])

        self.norm = norm_layer(dim)

    def forward(self, x, S, H, W):
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"
        x = x.reshape([B, S, H, W, C])
        x = F.gelu(x)
        x = self.norm(x)
        x = x.transpose([0, 4, 1, 2, 3])
        x = self.reduction(x)
        x = x.transpose([0, 2, 3, 4, 1]).reshape((B, -1, 2 * C))
        return x


class PatchExpanding(nn.Layer):

    def __init__(self, dim, norm_layer=nn.LayerNorm, tag=None):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        assert tag in [0, 1,
                       2], "Invalid `tag` value! `tag` must be 0, 1, or 2."
        if tag == 0:
            self.up = nn.Conv3DTranspose(dim, dim // 2, [1, 2, 2], [1, 2, 2])
        elif tag == 1:
            self.up = nn.Conv3DTranspose(dim, dim // 2, [2, 2, 2], [2, 2, 2])
        elif tag == 2:
            self.up = nn.Conv3DTranspose(dim,
                                         dim // 2, [2, 2, 2], [2, 2, 2],
                                         output_padding=[1, 0, 0])

    def forward(self, x, S, H, W):
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"

        x = x.reshape((B, S, H, W, C))

        x = self.norm(x)
        x = x.transpose((0, 4, 1, 2, 3))
        x = self.up(x)
        x = x.transpose((0, 2, 3, 4, 1)).reshape((B, -1, C // 2))

        return x


class MLP(nn.Layer):

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


class DropPath(nn.Layer):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = paddle.to_tensor(keep_prob, dtype='float32')
        shape = (inputs.shape[0], ) + (1, ) * (inputs.ndim - 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor()
        output = inputs.divide(keep_prob) * random_tensor
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


def window_partition(x, window_size):
    B, S, H, W, C = x.shape
    x = x.reshape([
        B, S // window_size[0], window_size[0], H // window_size[1],
        window_size[1], W // window_size[2], window_size[2], C
    ])
    x = x.transpose([0, 1, 3, 5, 2, 4, 6, 7])
    x = x.reshape([-1, window_size[0], window_size[1], window_size[2], C])
    return x


def window_reverse(windows, window_size, S, H, W):
    B = int(windows.shape[0] /
            (S * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.reshape(
        (B, S // window_size[0], H // window_size[1], W // window_size[2],
         window_size[0], window_size[1], window_size[2], -1))
    x = x.transpose((0, 1, 4, 2, 5, 3, 6, 7)).reshape((B, S, H, W, -1))
    return x


class WindowAttention_kv(nn.Layer):

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = paddle.create_parameter(
            shape=[(2 * window_size[0] - 1) * (2 * window_size[1] - 1) *
                   (2 * window_size[2] - 1), num_heads],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

        # Get pair-wise relative position index for each token inside the window
        coords_s = paddle.arange(self.window_size[0])
        coords_h = paddle.arange(self.window_size[1])
        coords_w = paddle.arange(self.window_size[2])
        coords = paddle.stack(paddle.meshgrid([coords_s, coords_h, coords_w]))
        coords_flatten = paddle.flatten(coords, 1)
        relative_coords = coords_flatten.unsqueeze(
            2) - coords_flatten.unsqueeze(1)
        relative_coords = relative_coords.transpose((1, 2, 0))
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] -
                                     1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(axis=-1)

    def get_relative_pos_bias_from_pos_index(self):
        table = self.relative_position_bias_table  # N x num_heads
        index = self.relative_position_index.reshape([-1])
        # NOTE: paddle does NOT support indexing Tensor by a Tensor
        relative_position_bias = paddle.index_select(x=table, index=index)
        return relative_position_bias

    def forward(self, skip, x_up, mask=None):
        B_, N, C = skip.shape

        kv = self.kv(skip)
        q = x_up

        kv = kv.reshape(
            (B_, N, 2, self.num_heads, C // self.num_heads)).transpose(
                (2, 0, 3, 1, 4))
        q = q.reshape((B_, N, self.num_heads, C // self.num_heads)).transpose(
            (0, 2, 1, 3))
        k, v = kv[0], kv[1]
        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        relative_position_bias = self.get_relative_pos_bias_from_pos_index()
        relative_position_bias = relative_position_bias.reshape([
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1
        ])
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape((B_ // nW, nW, self.num_heads, N,
                                 N)) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape((-1, self.num_heads, N, N))
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = paddle.matmul(attn, v).transpose([0, 2, 1, 3]).reshape([B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttention(nn.Layer):

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = paddle.create_parameter(
            shape=[(2 * window_size[0] - 1) * (2 * window_size[1] - 1) *
                   (2 * window_size[2] - 1), num_heads],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02))

        # Get pair-wise relative position index for each token inside the window
        coords_s = paddle.arange(self.window_size[0])
        coords_h = paddle.arange(self.window_size[1])
        coords_w = paddle.arange(self.window_size[2])
        coords = paddle.stack(paddle.meshgrid([coords_s, coords_h, coords_w]))
        coords_flatten = paddle.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:,
                                                                      None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(axis=-1)

    def get_relative_pos_bias_from_pos_index(self):
        table = self.relative_position_bias_table  # N x num_heads
        # index is a tensor
        index = self.relative_position_index.reshape(
            [-1])  # window_h*window_w * window_h*window_w
        # NOTE: paddle does NOT support indexing Tensor by a Tensor
        relative_position_bias = paddle.index_select(x=table, index=index)
        return relative_position_bias

    def forward(self, x, mask=None, pos_embed=None):
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape([B_, N, 3, self.num_heads,
                           C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        #qkv.shape:[B_,N,3*C]
        relative_position_bias = self.get_relative_pos_bias_from_pos_index()
        relative_position_bias = relative_position_bias.reshape([
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1
        ])
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(
                [x.shape[0] // nW, nW, self.num_heads, x.shape[1], x.shape[1]])
            attn += mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, x.shape[1], x.shape[1]])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = paddle.matmul(attn, v)
        x = x.transpose([0, 2, 1, 3])
        x = x.reshape([B_, N, C])
        if pos_embed is not None:
            x = x + pos_embed
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock_kv(nn.Layer):

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if tuple(self.input_resolution) == tuple(self.window_size):
            self.shift_size = [0, 0, 0]

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_kv(dim,
                                       window_size=self.window_size,
                                       num_heads=num_heads,
                                       qkv_bias=qkv_bias,
                                       qk_scale=qk_scale,
                                       attn_drop=attn_drop,
                                       proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x, mask_matrix, skip=None, x_up=None):
        assert self.shift_size == [0, 0, 0]
        B, L, C = x.shape
        S, H, W = self.input_resolution
        assert L == S * H * W, "input feature has wrong size"

        shortcut = x
        skip = self.norm1(skip)
        x_up = self.norm1(x_up)

        skip = skip.reshape((B, S, H, W, C))
        x_up = x_up.reshape((B, S, H, W, C))

        # Pad feature maps to multiples of window size
        pad_r = (self.window_size[2] -
                 W % self.window_size[2]) % self.window_size[2]
        pad_b = (self.window_size[1] -
                 H % self.window_size[1]) % self.window_size[1]
        pad_g = (self.window_size[0] -
                 S % self.window_size[0]) % self.window_size[0]

        skip = F.pad(skip, (0, pad_r, 0, pad_b, 0, pad_g), data_format="NDHWC")
        _, Sp, Hp, Wp, _ = skip.shape
        x_up = F.pad(x_up, (0, pad_r, 0, pad_b, 0, pad_g), data_format="NDHWC")

        skip = window_partition(skip, self.window_size)
        skip = skip.reshape((-1, self.window_size[0] * self.window_size[1] *
                             self.window_size[2], C))
        x_up = window_partition(x_up, self.window_size)
        x_up = x_up.reshape((-1, self.window_size[0] * self.window_size[1] *
                             self.window_size[2], C))
        # W-MSA/SW-MSA
        attn_windows = self.attn(skip, x_up)

        attn_windows = attn_windows.reshape(
            (-1, self.window_size[0], self.window_size[1], self.window_size[2],
             C))
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp,
                                   Wp)  # B H' W' C

        if min(self.shift_size) > 0:
            x = paddle.roll(shifted_x,
                            shifts=(self.shift_size[0], self.shift_size[1],
                                    self.shift_size[2]),
                            axis=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :]

        x = x.reshape((B, S * H * W, C))

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformerBlock(nn.Layer):

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if tuple(self.input_resolution) == tuple(self.window_size):
            # If window size is larger than input resolution, we don't partition windows
            self.shift_size = [0, 0, 0]
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim,
                                    window_size=self.window_size,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        S, H, W = self.input_resolution
        assert L == S * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape([B, S, H, W, C])

        pad_r = (self.window_size[2] -
                 W % self.window_size[2]) % self.window_size[2]
        pad_b = (self.window_size[1] -
                 H % self.window_size[1]) % self.window_size[1]
        pad_g = (self.window_size[0] -
                 S % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (0, pad_r, 0, pad_b, 0, pad_g), data_format="NDHWC")
        _, Sp, Hp, Wp, _ = x.shape

        if min(self.shift_size) > 0:
            shifted_x = paddle.roll(x,
                                    shifts=[
                                        -self.shift_size[0],
                                        -self.shift_size[1], -self.shift_size[2]
                                    ],
                                    axis=[1, 2, 3])
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(
            (-1,
             self.window_size[0] * self.window_size[1] * self.window_size[2],
             C))

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask, pos_embed=None)

        attn_windows = attn_windows.reshape(
            (-1, self.window_size[0], self.window_size[1], self.window_size[2],
             C))
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp,
                                   Wp)  # B H' W' C

        # Reverse cyclic shift
        if min(self.shift_size) > 0:
            x = paddle.roll(shifted_x,
                            shifts=[
                                self.shift_size[0], self.shift_size[1],
                                self.shift_size[2]
                            ],
                            axis=[1, 2, 3])
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :]

        x = x.reshape((B, S * H * W, C))

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Layer):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 i_layer=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = [
            window_size[0] // 2, window_size[1] // 2, window_size[2] // 2
        ]
        self.depth = depth
        self.i_layer = i_layer

        self.blocks = nn.LayerList([
            SwinTransformerBlock(dim=dim,
                                 input_resolution=input_resolution,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=[0, 0, 0] if
                                 (i % 2 == 0) else self.shift_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer) for i in range(depth)
        ])

        if downsample is not None:

            if i_layer == 1:
                self.downsample = downsample(dim=dim,
                                             norm_layer=norm_layer,
                                             tag=1)
            elif i_layer == 2:
                self.downsample = downsample(dim=dim,
                                             norm_layer=norm_layer,
                                             tag=2)
            elif i_layer == 0:
                self.downsample = downsample(dim=dim,
                                             norm_layer=norm_layer,
                                             tag=0)
            else:
                self.downsample = None
        else:
            self.downsample = None

    def forward(self, x, S, H, W):

        attn_mask = None
        for blk in self.blocks:
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, S, H, W)
            if self.i_layer != 1 and self.i_layer != 2:
                Ws, Wh, Ww = S, (H + 1) // 2, (W + 1) // 2
            else:
                Ws, Wh, Ww = S // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww
        else:
            return x, S, H, W, x, S, H, W


class BasicLayer_up(nn.Layer):

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=True,
                 i_layer=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = [
            window_size[0] // 2, window_size[1] // 2, window_size[2] // 2
        ]
        self.depth = depth

        self.blocks = nn.LayerList()
        self.blocks.append(
            SwinTransformerBlock_kv(dim=dim,
                                    input_resolution=input_resolution,
                                    num_heads=num_heads,
                                    window_size=window_size,
                                    shift_size=[0, 0, 0],
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop,
                                    attn_drop=attn_drop,
                                    drop_path=drop_path[0] if isinstance(
                                        drop_path, list) else drop_path,
                                    norm_layer=norm_layer))
        for i in range(depth - 1):
            self.blocks.append(
                SwinTransformerBlock(dim=dim,
                                     input_resolution=input_resolution,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     shift_size=self.shift_size,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop=drop,
                                     attn_drop=attn_drop,
                                     drop_path=drop_path[i + 1] if isinstance(
                                         drop_path, list) else drop_path,
                                     norm_layer=norm_layer))

        self.i_layer = i_layer
        if i_layer == 1:
            self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer, tag=1)
        elif i_layer == 0:
            self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer, tag=2)
        else:
            self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer, tag=0)

    def forward(self, x, skip, S, H, W):
        x_up = self.Upsample(x, S, H, W)
        x = skip + x_up
        if self.i_layer == 1:
            S, H, W = S * 2, H * 2, W * 2
        elif self.i_layer == 0:
            S, H, W = (S * 2) + 1, H * 2, W * 2
        else:
            S, H, W = S, H * 2, W * 2
        attn_mask = None
        x = self.blocks[0](x, attn_mask, skip=skip, x_up=x_up)
        for i in range(self.depth - 1):
            x = self.blocks[i + 1](x, attn_mask)

        return x, S, H, W


class Project(nn.Layer):

    def __init__(self,
                 in_dim,
                 out_dim,
                 stride,
                 padding,
                 activate,
                 norm,
                 last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv3D(in_dim,
                               out_dim,
                               kernel_size=3,
                               stride=stride,
                               padding=padding)
        self.conv2 = nn.Conv3D(out_dim,
                               out_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        Ws, Wh, Ww = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.norm1(x)
        x = x.transpose([0, 2, 1]).reshape((-1, self.out_dim, Ws, Wh, Ww))
        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            Ws, Wh, Ww = x.shape[2], x.shape[3], x.shape[4]
            x = x.flatten(2).transpose([0, 2, 1])
            x = self.norm2(x)
            x = x.transpose([0, 2, 1]).reshape((-1, self.out_dim, Ws, Wh, Ww))
        return x


class PatchEmbed(nn.Layer):

    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, norm_layer=None):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = tuple(repeat(patch_size, 3))
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        stride1 = [1, patch_size[1] // 2, patch_size[2] // 2]
        stride2 = [1, patch_size[1] // 2, patch_size[2] // 2]
        self.proj1 = Project(in_chans, embed_dim // 2, stride1, 1, nn.GELU,
                             nn.LayerNorm, False)
        self.proj2 = Project(embed_dim // 2, embed_dim, stride2, 1, nn.GELU,
                             nn.LayerNorm, True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, S, H, W = x.shape
        if W % self.patch_size[2] != 0:
            x = F.pad(
                x, [0, self.patch_size[2] - W % self.patch_size[2], 0, 0, 0, 0],
                data_format='NCDHW')
        if H % self.patch_size[1] != 0:
            x = F.pad(
                x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1], 0, 0),
                data_format='NCDHW')
        if S % self.patch_size[0] != 0:
            x = F.pad(
                x, (0, 0, 0, 0, 0, self.patch_size[0] - S % self.patch_size[0]),
                data_format='NCDHW')
        x = self.proj1(x)  # B C Ws Wh Ww
        x = self.proj2(x)  # B C Ws Wh Ww
        if self.norm is not None:
            Ws, Wh, Ww = x.shape[2], x.shape[3], x.shape[4]
            x = x.flatten(2).transpose([0, 2, 1])
            x = self.norm(x)
            x = x.transpose([0, 2, 1]).reshape((-1, self.embed_dim, Ws, Wh, Ww))
        return x


class Encoder(nn.Layer):

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=1,
                 embed_dim=96,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 down_stride=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3)):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.layers = nn.LayerList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(pretrain_img_size[0] //
                                  down_stride[i_layer][0],
                                  pretrain_img_size[1] //
                                  down_stride[i_layer][1],
                                  pretrain_img_size[2] //
                                  down_stride[i_layer][2]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if
                (i_layer < self.num_layers - 1) else None,
                i_layer=i_layer)
            self.layers.append(layer)
        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features
        # Add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_sublayer(layer_name, layer)

    def forward(self, x):
        x = self.patch_embed(x)
        down = []

        Ws, Wh, Ww = x.shape[2], x.shape[3], x.shape[4]

        x = x.flatten(2).transpose([0, 2, 1])
        x = self.pos_drop(x)

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, S, H, W, x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.reshape(
                    (-1, S, H, W, self.num_features[i])).transpose(
                        (0, 4, 1, 2, 3))

                down.append(out)
        return down


class Decoder(nn.Layer):

    def __init__(self,
                 pretrain_img_size,
                 embed_dim,
                 patch_size=4,
                 depths=[2, 2, 2],
                 num_heads=[24, 12, 6],
                 window_size=4,
                 up_stride=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()

        # Build layers
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers)[::-1]:
            layer = BasicLayer_up(
                dim=int(embed_dim * 2**(len(depths) - i_layer - 1)),
                input_resolution=(pretrain_img_size[0] // up_stride[i_layer][0],
                                  pretrain_img_size[1] // up_stride[i_layer][1],
                                  pretrain_img_size[2] //
                                  up_stride[i_layer][2]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpanding,
                i_layer=i_layer)
            self.layers.append(layer)
        self.num_features = [
            int(embed_dim * 2**i) for i in range(self.num_layers)
        ]

    def forward(self, x, skips):

        outs = []
        S, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose([0, 2, 1])
        for index, i in enumerate(skips):
            i = i.flatten(2).transpose([0, 2, 1])
            skips[index] = i
        x = self.pos_drop(x)

        for i in range(self.num_layers)[::-1]:
            layer = self.layers[i]

            x, S, H, W, = layer(x, skips[i], S, H, W)
            out = x.reshape((-1, S, H, W, self.num_features[i]))
            outs.append(out)
        return outs


class final_patch_expanding(nn.Layer):

    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        self.up = nn.Conv3DTranspose(dim, num_class, patch_size, patch_size)

    def forward(self, x):
        x = x.transpose((0, 4, 1, 2, 3))
        x = self.up(x)

        return x


@manager.MODELS.add_component
class nnFormer(nn.Layer):
    """
    The nnformer implementation based on PaddlePaddle.

    The original article refers to
    Hong-Yu Zhou, et al. "nnFormer: Volumetric Medical Image Segmentation via a 3D Transformer"
    (https://arxiv.org/pdf/2109.03201.pdf).

    Args:
        crop_size (list): The input size of image.
        embedding_dim (int): The number of embedding dimensions.
        input_channels (int): The channels of input image.
        num_classes (int): The number of classes.
        conv_op (nn.Layer): The convolution operator to use.
        depths (list): The number of self-attention blocks in each stage.
        num_heads (list): The number of heads to use in multi-head attention.
        patch_size (list): The patch_size of each patch embedding.
        window_size (list): The size of windows when performing self-attention (LV-MSA).
        down_stride (list): The stride in each dimension when performing downsampling.
        deep_supervision (bool) : Whether to adopt deep supervision.
    """

    def __init__(self,
                 crop_size=[14, 160, 160],
                 embedding_dim=96,
                 input_channels=1,
                 num_classes=4,
                 conv_op=nn.Conv3D,
                 depths=[2, 2, 2, 2],
                 num_heads=[3, 6, 12, 24],
                 patch_size=[2, 4, 4],
                 window_size=[[3, 5, 5], [3, 5, 5], [7, 10, 10], [3, 5, 5]],
                 down_stride=[[1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2]],
                 deep_supervision=True):
        super(nnFormer, self).__init__()
        self.img_shape = crop_size
        self.data_format = 'NCDHW'
        self.do_ds = deep_supervision
        self.num_classes = num_classes
        self.conv_op = conv_op

        self.upscale_logits_ops = []

        self.upscale_logits_ops.append(lambda x: x)
        embed_dim = embedding_dim
        depths = depths
        num_heads = num_heads
        patch_size = patch_size
        window_size = window_size
        down_stride = down_stride

        self.model_down = Encoder(pretrain_img_size=crop_size,
                                  window_size=window_size,
                                  embed_dim=embed_dim,
                                  patch_size=patch_size,
                                  depths=depths,
                                  num_heads=num_heads,
                                  in_chans=input_channels,
                                  down_stride=down_stride)
        self.decoder = Decoder(pretrain_img_size=crop_size,
                               embed_dim=embed_dim,
                               window_size=window_size[::-1][1:],
                               patch_size=patch_size,
                               num_heads=num_heads[::-1][1:],
                               depths=depths[::-1][1:],
                               up_stride=down_stride[::-1][1:])

        self.final = []
        if self.do_ds:

            for i in range(len(depths) - 1):
                self.final.append(
                    final_patch_expanding(embed_dim * 2**i,
                                          num_classes,
                                          patch_size=patch_size))

        else:
            self.final.append(
                final_patch_expanding(embed_dim,
                                      num_classes,
                                      patch_size=patch_size))

        self.final = nn.LayerList(self.final)

    def forward(self, x):

        seg_outputs = []
        skips = self.model_down(x)
        neck = skips[-1]

        out = self.decoder(neck, skips)

        if self.do_ds:
            for i in range(len(out)):
                out_put = F.interpolate(self.final[-(i + 1)](out[i]),
                                        size=x.shape[2:],
                                        data_format='NCDHW',
                                        mode='trilinear')

                seg_outputs.append(out_put)
            return seg_outputs[::-1]
        else:
            seg_outputs.append(self.final[0](out[-1]))
            return seg_outputs


if __name__ == "__main__":
    embedding_dim = 96
    depths = [2, 2, 2, 2]
    num_heads = [3, 6, 12, 24]
    embedding_patch_size = [1, 4, 4]
    window_size = [[3, 5, 5], [3, 5, 5], [7, 10, 10], [3, 5, 5]]
    down_stride = [[1, 4, 4], [1, 8, 8], [2, 16, 16], [4, 32, 32]]
    net = nnFormer(crop_size=[14, 160, 160],
                   embedding_dim=embedding_dim,
                   input_channels=1,
                   num_classes=2,
                   conv_op=nn.Conv3D,
                   depths=depths,
                   num_heads=num_heads,
                   patch_size=embedding_patch_size,
                   window_size=window_size,
                   down_stride=down_stride,
                   deep_supervision=True)

    input = paddle.rand([1, 1, 14, 160, 160])
    out = net(input)
    for index, i in enumerate(out):
        print("{} out.shape:{}".format(index, i.shape))

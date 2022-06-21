# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/tamasino52/UNETR/blob/main/unetr.py

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

from typing import List, Tuple

import numpy as np
import paddle
import paddle.nn as nn
from paddle import Tensor

from medicalseg.cvlibs import manager


# green block in Fig.1
class TranspConv3DBlock(nn.Layer):
    def __init__(self, in_planes, out_planes):
        super(TranspConv3DBlock, self).__init__()
        self.block = nn.Conv3DTranspose(
            in_planes,
            out_planes,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            bias_attr=False)

    def forward(self, x):
        y = self.block(x)
        return y


class TranspConv3DConv3D(nn.Layer):
    def __init__(self, in_planes, out_planes, layers=1, conv_block=False):
        """
        blue box in Fig.1
        Args:
            in_planes: in channels of transpose convolution
            out_planes: out channels of transpose convolution
            layers: number of blue blocks, transpose convs
            conv_block: whether to include a conv block after each transpose conv. deafaults to False
        """
        super(TranspConv3DConv3D, self).__init__()
        self.blocks = nn.LayerList([TranspConv3DBlock(in_planes, out_planes), ])
        if conv_block:
            self.blocks.append(
                Conv3DBlock(
                    out_planes, out_planes, double=False))

        if int(layers) >= 2:
            for _ in range(int(layers) - 1):
                self.blocks.append(TranspConv3DBlock(out_planes, out_planes))
                if conv_block:
                    self.blocks.append(
                        Conv3DBlock(
                            out_planes, out_planes, double=False))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# yellow block in Fig.1
class Conv3DBlock(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 double=True,
                 norm=nn.BatchNorm3D,
                 skip=True):
        super(Conv3DBlock, self).__init__()

        self.skip = skip
        self.downsample = in_planes != out_planes
        self.final_activation = nn.LeakyReLU(negative_slope=0.01)
        padding = (kernel_size - 1) // 2
        if double:
            self.conv_block = nn.Sequential(
                nn.Conv3D(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding),
                norm(out_planes),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Conv3D(
                    out_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding),
                norm(out_planes))
        else:
            self.conv_block = nn.Sequential(
                nn.Conv3D(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding),
                norm(out_planes))

        if self.skip and self.downsample:
            self.conv_down = nn.Sequential(
                nn.Conv3D(
                    in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                norm(out_planes))

    def forward(self, x):
        y = self.conv_block(x)
        if self.skip:
            res = x
            if self.downsample:
                res = self.conv_down(res)
            y = y + res
        return self.final_activation(y)


class AbsPositionalEncoding1D(nn.Layer):
    def __init__(self, tokens, dim):
        super(AbsPositionalEncoding1D, self).__init__()
        params = paddle.randn(shape=[1, tokens, dim])
        self.abs_pos_enc = paddle.create_parameter(
            shape=params.shape,
            dtype=str(params.numpy().dtype),
            default_initializer=paddle.nn.initializer.Assign(params))

    def forward(self, x):

        batch = x.shape[0]

        tile = batch // self.abs_pos_enc.shape[0]
        expb = paddle.tile(self.abs_pos_enc, repeat_times=(tile, 1, 1))

        return x + expb


class Embeddings3D(nn.Layer):
    def __init__(self,
                 input_dim,
                 embed_dim,
                 cube_size,
                 patch_size=16,
                 dropout=0.1):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) /
                             (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3D(
            in_channels=input_dim,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias_attr=False)
        self.position_embeddings = AbsPositionalEncoding1D(self.n_patches,
                                                           embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x is a 5D tensor
        """
        patch_embeddings = self.patch_embeddings(x)
        shape = patch_embeddings.shape
        _d = paddle.reshape(patch_embeddings, [shape[0], shape[1], -1])
        _d = paddle.transpose(_d, perm=[0, 2, 1])
        embeddings = self.dropout(self.position_embeddings(_d))
        return embeddings


def compute_mhsa(q, k, v, scale_factor=1, mask=None):
    # resulted shape will be: [batch, heads, tokens, tokens]

    k = paddle.transpose(k, perm=[0, 1, 3, 2])
    scaled_dot_prod = paddle.matmul(q, k) * scale_factor

    if mask is not None:
        assert mask.shape == scaled_dot_prod.shape[2:]
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

    attention = paddle.nn.functional.softmax(scaled_dot_prod, axis=-1)
    # calc result per head

    return paddle.matmul(attention, v)


class MultiHeadSelfAttention(nn.Layer):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias_attr=False)
        self.W_0 = nn.Linear(_dim, dim, bias_attr=False)
        self.scale_factor = self.dim_head**-0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]
        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]

        shape = qkv.shape
        k = 3
        h = self.heads

        d = paddle.reshape(qkv, [shape[0], shape[1], -1, k, h])
        d = paddle.transpose(d, perm=[3, 0, 4, 1, 2])

        q, k, v = tuple(d)

        out = compute_mhsa(q, k, v, mask=mask, scale_factor=self.scale_factor)

        # re-compose: merge heads with dim_head

        out = paddle.flatten(
            paddle.transpose(
                out, perm=[0, 2, 1, 3]), start_axis=2, stop_axis=3)

        return self.W_0(out)


class TransformerBlock(nn.Layer):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    """

    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=None,
                 dim_linear_block=1024,
                 dropout=0.1,
                 activation=nn.GELU,
                 mhsa=None,
                 prenorm=False):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mhsa: if provided you can change the vanilla self-attention block
            prenorm: if the layer norm will be applied before the mhsa or after
        """
        super().__init__()
        self.mhsa = mhsa if mhsa is not None else MultiHeadSelfAttention(
            dim=dim, heads=heads, dim_head=dim_head)
        self.prenorm = prenorm
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout))

    def forward(self, x, mask=None):
        if self.prenorm:
            y = self.drop(self.mhsa(self.norm_1(x), mask)) + x
            out = self.linear(self.norm_2(y)) + y
        else:
            y = self.norm_1(self.drop(self.mhsa(x, mask)) + x)
            out = self.norm_2(self.linear(y) + y)
        return out


class TransformerEncoder(nn.Layer):
    def __init__(self, embed_dim, num_heads, num_layers, dropout,
                 extract_layers, dim_linear_block):
        super().__init__()
        self.layer = nn.LayerList()
        self.extract_layers = extract_layers

        # makes TransformerBlock device aware
        self.block_list = nn.LayerList()
        for _ in range(num_layers):
            self.block_list.append(
                TransformerBlock(
                    dim=embed_dim,
                    heads=num_heads,
                    dim_linear_block=dim_linear_block,
                    dropout=dropout,
                    prenorm=True))

    def forward(self, x):
        extract_layers = []
        for depth, layer_block in enumerate(self.block_list):
            x = layer_block(x)
            if (depth + 1) in self.extract_layers:
                extract_layers.append(x)

        return extract_layers


# based on https://arxiv.org/abs/2103.10504
@manager.MODELS.add_component
class UNETR(nn.Layer):
    def __init__(self,
                 img_shape=(128, 128, 128),
                 in_channels=4,
                 embed_dim=768,
                 patch_size=16,
                 num_heads=12,
                 dropout=0.0,
                 ext_layers=[3, 6, 9, 12],
                 norm='instance',
                 base_filters=16,
                 dim_linear_block=3072,
                 num_classes=4):
        """
        The UNETR implementation based on PaddlePaddle.
        The original article refers to
        Ali Hatamizadeh, Yucheng Tang, Vishwesh Nath, Dong Yang, Andriy Myronenko, Bennett Landman, Holger Roth, Daguang Xu
        UNETR: Transformers for 3D Medical Image Segmentation
        (https://arxiv.org/abs/2103.10504)
        Implementation of this model is borrowed and modified(from torch to paddle) from here:
        https://github.com/tamasino52/UNETR/blob/main/unetr.py

        Args:
            img_shape: volume shape, provided as a tuple
            in_channels: input modalities/channels
            num_classes: number of classes
            embed_dim: transformer embed dim.
            patch_size: the non-overlapping patches to be created
            num_heads: for the transformer encoder
            dropout: percentage for dropout
            ext_layers: transformer layers to use their output
            version: 'light' saves some parameters in the decoding part
            norm: batch or instance norm for the conv blocks
        """
        super().__init__()
        self.num_layers = 12
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.img_shape = img_shape if type(img_shape) == 'tuple' else eval(
            img_shape)
        img_shape = self.img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.ext_layers = ext_layers
        self.patch_dim = [int(x / patch_size) for x in img_shape]

        self.norm = nn.BatchNorm3d if norm == 'batch' else nn.InstanceNorm3D

        self.embed = Embeddings3D(
            input_dim=in_channels,
            embed_dim=embed_dim,
            cube_size=img_shape,
            patch_size=patch_size,
            dropout=dropout)

        self.transformer = TransformerEncoder(
            embed_dim,
            num_heads,
            self.num_layers,
            dropout,
            ext_layers,
            dim_linear_block=dim_linear_block)

        self.init_conv = Conv3DBlock(
            in_channels, base_filters, double=True, norm=self.norm)

        # blue blocks in Fig.1
        self.z3_blue_conv = TranspConv3DConv3D(
            in_planes=embed_dim, out_planes=base_filters * 2, layers=3)

        self.z6_blue_conv = TranspConv3DConv3D(
            in_planes=embed_dim, out_planes=base_filters * 4, layers=2)

        self.z9_blue_conv = TranspConv3DConv3D(
            in_planes=embed_dim, out_planes=base_filters * 8, layers=1)

        # Green blocks in Fig.1
        self.z12_deconv = TranspConv3DBlock(embed_dim, base_filters * 8)

        self.z9_deconv = TranspConv3DBlock(base_filters * 8, base_filters * 4)
        self.z6_deconv = TranspConv3DBlock(base_filters * 4, base_filters * 2)
        self.z3_deconv = TranspConv3DBlock(base_filters * 2, base_filters)

        # Yellow blocks in Fig.1
        self.z9_conv = Conv3DBlock(
            base_filters * 8 * 2, base_filters * 8, double=True, norm=self.norm)
        self.z6_conv = Conv3DBlock(
            base_filters * 4 * 2, base_filters * 4, double=True, norm=self.norm)
        self.z3_conv = Conv3DBlock(
            base_filters * 2 * 2, base_filters * 2, double=True, norm=self.norm)
        # out convolutions
        self.out_conv = nn.Sequential(
            # last yellow conv block
            Conv3DBlock(
                base_filters * 2, base_filters, double=True, norm=self.norm),
            # grey block, final classification layer
            nn.Conv3D(
                base_filters, num_classes, kernel_size=1, stride=1))

    def forward(self, x):
        transf_input = self.embed(x)

        def rearr(t):
            v_x, v_y, v_z = self.patch_dim
            shape = t.shape
            d = paddle.reshape(t, [shape[0], v_x, v_y, v_z, shape[-1]])
            d = paddle.transpose(d, perm=[0, 4, 1, 2, 3])
            return d

        z3, z6, z9, z12 = map(rearr, self.transformer(transf_input))

        # Blue convs
        z0 = self.init_conv(x)
        z3 = self.z3_blue_conv(z3)
        z6 = self.z6_blue_conv(z6)
        z9 = self.z9_blue_conv(z9)

        # Green block for z12
        z12 = self.z12_deconv(z12)
        # Concat + yellow conv
        y = paddle.concat([z12, z9], axis=1)
        y = self.z9_conv(y)

        # Green block for z6
        y = self.z9_deconv(y)
        # Concat + yellow conv
        y = paddle.concat([y, z6], axis=1)
        y = self.z6_conv(y)

        # Green block for z3
        y = self.z6_deconv(y)
        # Concat + yellow conv
        y = paddle.concat([y, z3], axis=1)
        y = self.z3_conv(y)

        y = self.z3_deconv(y)
        y = paddle.concat([y, z0], axis=1)
        return (self.out_conv(y), )

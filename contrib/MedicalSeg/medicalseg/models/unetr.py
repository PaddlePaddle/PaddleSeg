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

import paddle
import paddle.nn as nn

from medicalseg.cvlibs import manager

from .unetr_module import TranspConv3DBlock, BlueBlock, Conv3DBlock
from .unetr_volume_embedding import Embeddings3D
from .unetr_vanilla import TransformerBlock


class TransformerEncoder(nn.Layer):
    def __init__(self, embed_dim, num_heads, num_layers, dropout, extract_layers, dim_linear_block):
        super().__init__()
        self.layer = nn.LayerList()
        self.extract_layers = extract_layers

        # makes TransformerBlock device aware
        self.block_list = nn.LayerList()
        for _ in range(num_layers):
            self.block_list.append(TransformerBlock(dim=embed_dim, heads=num_heads,
                                                    dim_linear_block=dim_linear_block, dropout=dropout, prenorm=True))

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
    def __init__(self, img_shape=(128, 128, 128), input_dim=4, output_dim=3,
                 embed_dim=768, patch_size=16, num_heads=12, dropout=0.0,
                 ext_layers=[3, 6, 9, 12], norm='instance',
                 base_filters=16,
                 dim_linear_block=3072, num_classes=4):
        """
        Args:
            img_shape: volume shape, provided as a tuple
            input_dim: input modalities/channels
            output_dim: number of classes
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
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape if type(img_shape) == 'tuple' else eval(img_shape)
        img_shape = self.img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.ext_layers = ext_layers
        self.patch_dim = [int(x / patch_size) for x in img_shape]

        self.norm = nn.BatchNorm3d if norm == 'batch' else nn.InstanceNorm3D

        self.embed = Embeddings3D(input_dim=input_dim, embed_dim=embed_dim,
                                  cube_size=img_shape, patch_size=patch_size, dropout=dropout)

        self.transformer = TransformerEncoder(embed_dim, num_heads,
                                              self.num_layers, dropout, ext_layers,
                                              dim_linear_block=dim_linear_block)

        self.init_conv = Conv3DBlock(input_dim, base_filters, double=True, norm=self.norm)

        # blue blocks in Fig.1
        self.z3_blue_conv = BlueBlock(in_planes=embed_dim, out_planes=base_filters * 2, layers=3)

        self.z6_blue_conv = BlueBlock(in_planes=embed_dim, out_planes=base_filters * 4, layers=2)

        self.z9_blue_conv = BlueBlock(in_planes=embed_dim, out_planes=base_filters * 8, layers=1)

        # Green blocks in Fig.1
        self.z12_deconv = TranspConv3DBlock(embed_dim, base_filters * 8)

        self.z9_deconv = TranspConv3DBlock(base_filters * 8, base_filters * 4)
        self.z6_deconv = TranspConv3DBlock(base_filters * 4, base_filters * 2)
        self.z3_deconv = TranspConv3DBlock(base_filters * 2, base_filters)

        # Yellow blocks in Fig.1
        self.z9_conv = Conv3DBlock(base_filters * 8 * 2, base_filters * 8, double=True, norm=self.norm)
        self.z6_conv = Conv3DBlock(base_filters * 4 * 2, base_filters * 4, double=True, norm=self.norm)
        self.z3_conv = Conv3DBlock(base_filters * 2 * 2, base_filters * 2, double=True, norm=self.norm)
        # out convolutions
        self.out_conv = nn.Sequential(
            # last yellow conv block
            Conv3DBlock(base_filters * 2, base_filters, double=True, norm=self.norm),
            # grey block, final classification layer
            nn.Conv3D(base_filters, output_dim, kernel_size=1, stride=1))

    def forward(self, x):
        transf_input = self.embed(x)

        v_x, v_y, v_z = self.patch_dim

        def rearr(t):
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
        return (self.out_conv(y),)
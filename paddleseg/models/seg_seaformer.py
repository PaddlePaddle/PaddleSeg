# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers


class FusionBlock(nn.Layer):
    def __init__(self, inp: int, oup: int, embed_dim: int):
        super().__init__()

        self.local_embedding = layers.ConvBN(
            inp, embed_dim, kernel_size=1, bias_attr=False)
        self.global_act = layers.ConvBN(
            oup, embed_dim, kernel_size=1, bias_attr=False)
        self.act = nn.Hardsigmoid()

    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(
            self.act(global_act),
            size=[H, W],
            mode='bilinear',
            align_corners=False)
        out = local_feat * sig_act
        return out


@manager.MODELS.add_component
class Seg_SeaFormer(nn.Layer):
    def __init__(self,
                 backbone,
                 in_channels=[64, 192, 256],
                 in_index=[0, 1, 2],
                 channels=160,
                 embed_dims=[128, 160],
                 num_classes=150,
                 is_dw=True,
                 dropout_ratio=0.1,
                 align_corners=False,
                 input_transform='multiple_select'):
        super().__init__()

        self.head_channels = channels
        self.backbone = backbone

        self.in_index = in_index
        self.input_transform = input_transform
        self.align_corners = align_corners

        self.embed_dims = embed_dims

        self.linear_fuse = layers.ConvBNReLU(
            self.head_channels,
            self.head_channels,
            1,
            stride=1,
            groups=self.head_channels if is_dw else 1,
            bias_attr=False)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2D(dropout_ratio)

        self.cls_seg = nn.Conv2D(channels, num_classes, kernel_size=1)

        for i in range(len(embed_dims)):
            fuse = FusionBlock(
                in_channels[0] if i == 0 else embed_dims[i - 1],
                in_channels[i + 1],
                embed_dim=embed_dims[i])
            setattr(self, f"fuse{i + 1}", fuse)

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        inputs = self.backbone(inputs)

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            xx = paddle.concat(upsampled_inputs, axis=1)

        elif self.input_transform == 'multiple_select':
            xx = [inputs[i] for i in self.in_index]
        else:
            xx = inputs[self.in_index]

        x_detail = xx[0]
        for i in range(len(self.embed_dims)):
            fuse = getattr(self, f"fuse{i + 1}")
            x_detail = fuse(x_detail, xx[i + 1])
        feat = self.linear_fuse(x_detail)

        if self.dropout is not None:
            feat = self.dropout(feat)

        x = self.cls_seg(feat)
        x = [
            F.interpolate(
                x,
                size=[H, W],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        return x

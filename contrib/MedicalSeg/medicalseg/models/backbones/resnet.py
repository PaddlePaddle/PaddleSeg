# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/Beckschen/TransUNet/blob/main/networks/vit_seg_modeling_resnet_skip.py

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

from os.path import join as pjoin

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from medicalseg.cvlibs import manager


class StdConv2d(nn.Conv2D):
    def forward(self, x):
        if self._padding_mode != 'zeros':
            x = F.pad(x,
                      self._reversed_padding_repeated_twice,
                      mode=self._padding_mode,
                      data_format=self._data_format)

        w = self.weight
        v = paddle.var(w, axis=[1, 2, 3], keepdim=True, unbiased=False)
        m = paddle.mean(w, axis=[1, 2, 3], keepdim=True)
        w = (w - m) / paddle.sqrt(v + 1e-5)

        out = F.conv._conv_nd(
            x,
            w,
            bias=self.bias,
            stride=self._stride,
            padding=self._updated_padding,
            padding_algorithm=self._padding_algorithm,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format,
            channel_dim=self._channel_dim,
            op_type=self._op_type,
            use_cudnn=self._use_cudnn)
        return out


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(
        cin,
        cout,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias_attr=bias,
        groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(
        cin, cout, kernel_size=1, stride=stride, padding=0, bias_attr=bias)


class Bottleneck(nn.Layer):
    """ResNet with GroupNorm and Weight Standardization."""

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, epsilon=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, epsilon=1e-6)
        self.conv2 = conv3x3(
            cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, epsilon=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU()

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


@manager.BACKBONES.add_component
class ResNet(nn.Layer):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(
            ('conv', StdConv2d(
                3, width, kernel_size=7, stride=2, bias_attr=False, padding=3)),
            ('gn', nn.GroupNorm(
                32, width, epsilon=1e-6)), ('relu', nn.ReLU()))

        self.body = nn.Sequential(
            ('block1', nn.Sequential(*([('unit1', Bottleneck(
                cin=width, cout=width * 4, cmid=width))] + [
                    (f'unit{i:d}', Bottleneck(
                        cin=width * 4, cout=width * 4, cmid=width))
                    for i in range(2, block_units[0] + 1)
                ]))),
            ('block2', nn.Sequential(*([('unit1', Bottleneck(
                cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] + [
                    (f'unit{i:d}', Bottleneck(
                        cin=width * 8, cout=width * 8, cmid=width * 2))
                    for i in range(2, block_units[1] + 1)
                ]))),
            ('block3', nn.Sequential(*([('unit1', Bottleneck(
                cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] + [
                    (f'unit{i:d}', Bottleneck(
                        cin=width * 16, cout=width * 16, cmid=width * 4))
                    for i in range(2, block_units[2] + 1)
                ]))), )

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.shape
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2D(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.shape[2] == right_size:
                feat = x
            else:
                feat = paddle.zeros((b, x.shape[1], right_size, right_size))
                feat[:, :, 0:x.shape[2], 0:x.shape[3]] = x[:]
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]

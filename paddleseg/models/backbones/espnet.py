# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers


class DownSampler(nn.Layer):
    """
    Down sampler.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        branches (int): Number of branches.
        kernel_size_maximum (int): A maximum value of kernel_size for EESP block.
        shortcut (bool): Use shortcut or not. Default: True.
    """
    def __init__(self, in_channels, out_channels, branches=4, kernel_size_maximum=9, shortcut=True):
        super().__init__()
        eesp_down_channels = out_channels - in_channels
        assert eesp_down_channels >= 0, "DownSampler"
        self.eesp = layers.EESP(in_channels,
                                eesp_down_channels,
                                stride=2,
                                branches=branches,
                                kernel_size_maximum=kernel_size_maximum,
                                down_method='avg')
        self.avg = nn.AvgPool2D(kernel_size=3, padding=1, stride=2)
        if shortcut:
            self.shortcut_layer = nn.Sequential(
                layers.ConvBNReLU(3, 3, 3, 1),
                layers.ConvBN(3, out_channels, 1, 1),
            )
        self._act = nn.PReLU()

    def forward(self, x, inputs=None):
        avg_out = self.avg(x)
        eesp_out = self.eesp(x)
        output = paddle.concat([avg_out, eesp_out], axis=1)

        if inputs is not None:
            w1 = avg_out.shape[2]
            while True:
                inputs = F.avg_pool2d(inputs, kernel_size=3, padding=1, stride=2)
                w2 = inputs.shape[2]
                if w2 == w1:
                    break
            output = output + self.shortcut_layer(inputs)
        return self._act(output)


@manager.BACKBONES.add_component
class EESPNet(nn.Layer):
    """
    The ESPNetV2 implementation based on PaddlePaddle.

    The original article refers to
    Sachin Mehta, Mohammad Rastegari, Linda Shapiro, and Hannaneh Hajishirzi. "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network"
    (https://arxiv.org/abs/1811.11431).

    Args:
        in_channels (int): Number of input channels.
        drop_prob (float): The probability of dropout.
        scale (float, optional): The scale of channels, only support scale <= 1.5 and scale == 2. Default: 1.0.
    """
    def __init__(self, in_channels=3, drop_prob=0.1, scale=1.0):
        super().__init__()
        reps = [0, 3, 7, 3]

        num_level = 4   # 1/2, 1/4, 1/8, 1/16
        kernel_size_limitations = [13, 11, 9, 7]     # kernel size limitation
        branch_list = [4] * len(kernel_size_limitations)    # branches at different levels

        base_channels = 32  # first conv output channels
        channels_config = [base_channels] * num_level

        channels = 0
        for i in range(num_level):
            if i == 0:
                channels = int(base_channels * scale)
                channels = math.ceil(channels / branch_list[0]) * branch_list[0]
                channels_config[i] = base_channels if channels > base_channels else channels
            else:
                channels_config[i] = channels * pow(2, i)

        self.level1 = layers.ConvBNPReLU(in_channels, channels_config[0], 3, 2)

        self.level2 = DownSampler(channels_config[0],
                                  channels_config[1],
                                  branches=branch_list[0],
                                  kernel_size_maximum=kernel_size_limitations[0],
                                  shortcut=True)

        self.level3_0 = DownSampler(channels_config[1],
                                    channels_config[2],
                                    branches=branch_list[1],
                                    kernel_size_maximum=kernel_size_limitations[1],
                                    shortcut=True)
        self.level3 = nn.LayerList()
        for i in range(reps[1]):
            self.level3.append(
                layers.EESP(
                    channels_config[2], channels_config[2], stride=1, branches=branch_list[2], kernel_size_maximum=kernel_size_limitations[2]
                )
            )

        self.level4_0 = DownSampler(channels_config[2],
                                    channels_config[3],
                                    branches=branch_list[2],
                                    kernel_size_maximum=kernel_size_limitations[2],
                                    shortcut=True)
        self.level4 = nn.LayerList()
        for i in range(reps[2]):
            self.level4.append(
                layers.EESP(
                    channels_config[3], channels_config[3], stride=1, branches=branch_list[3], kernel_size_maximum=kernel_size_limitations[3]
                )
            )

        self.out_channels = channels_config

        self.init_params()

    def init_params(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                param_init.kaiming_normal_init(m.weight)
                if m.bias is not None:
                    param_init.constant_init(m.bias, value=0.0)
            elif isinstance(m, nn.BatchNorm2D):
                param_init.constant_init(m.weight, value=1.0)
                param_init.constant_init(m.bias, value=0.0)
            elif isinstance(m, nn.Linear):
                param_init.normal_init(m.weight, std=0.001)
                if m.bias is not None:
                    param_init.constant_init(m.bias, value=0.0)

    def forward(self, x):
        out_l1 = self.level1(x)
        out_l2 = self.level2(out_l1, x)
        out_l3_0 = self.level3_0(out_l2, x)
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)
        out_l4_0 = self.level4_0(out_l3, x)
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)
        return out_l1, out_l2, out_l3, out_l4






































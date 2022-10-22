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

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D

from paddleseg.cvlibs import manager

import ppmatting


class ConvBlock(nn.Layer):
    def __init__(self, input_channels, output_channels, groups, name=None):
        super(ConvBlock, self).__init__()

        self.groups = groups
        self._conv_1 = Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=ParamAttr(name=name + "1_weights"),
            bias_attr=False)
        if groups == 2 or groups == 3 or groups == 4:
            self._conv_2 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(name=name + "2_weights"),
                bias_attr=False)
        if groups == 3 or groups == 4:
            self._conv_3 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(name=name + "3_weights"),
                bias_attr=False)
        if groups == 4:
            self._conv_4 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(name=name + "4_weights"),
                bias_attr=False)

        self._pool = MaxPool2D(
            kernel_size=2, stride=2, padding=0, return_mask=True)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        if self.groups == 2 or self.groups == 3 or self.groups == 4:
            x = self._conv_2(x)
            x = F.relu(x)
        if self.groups == 3 or self.groups == 4:
            x = self._conv_3(x)
            x = F.relu(x)
        if self.groups == 4:
            x = self._conv_4(x)
            x = F.relu(x)
        skip = x
        x, max_indices = self._pool(x)
        return x, max_indices, skip


class VGGNet(nn.Layer):
    def __init__(self, input_channels=3, layers=11, pretrained=None):
        super(VGGNet, self).__init__()
        self.pretrained = pretrained

        self.layers = layers
        self.vgg_configure = {
            11: [1, 1, 2, 2, 2],
            13: [2, 2, 2, 2, 2],
            16: [2, 2, 3, 3, 3],
            19: [2, 2, 4, 4, 4]
        }
        assert self.layers in self.vgg_configure.keys(), \
            "supported layers are {} but input layer is {}".format(
                self.vgg_configure.keys(), layers)
        self.groups = self.vgg_configure[self.layers]

        # matting的第一层卷积输入为4通道，初始化是直接初始化为0
        self._conv_block_1 = ConvBlock(
            input_channels, 64, self.groups[0], name="conv1_")
        self._conv_block_2 = ConvBlock(64, 128, self.groups[1], name="conv2_")
        self._conv_block_3 = ConvBlock(128, 256, self.groups[2], name="conv3_")
        self._conv_block_4 = ConvBlock(256, 512, self.groups[3], name="conv4_")
        self._conv_block_5 = ConvBlock(512, 512, self.groups[4], name="conv5_")

        # 这一层的初始化需要利用vgg fc6的参数转换后进行初始化，可以暂时不考虑初始化
        self._conv_6 = Conv2D(
            512, 512, kernel_size=3, padding=1, bias_attr=False)

        self.init_weight()

    def forward(self, inputs):
        fea_list = []
        ids_list = []
        x, ids, skip = self._conv_block_1(inputs)
        fea_list.append(skip)
        ids_list.append(ids)
        x, ids, skip = self._conv_block_2(x)
        fea_list.append(skip)
        ids_list.append(ids)
        x, ids, skip = self._conv_block_3(x)
        fea_list.append(skip)
        ids_list.append(ids)
        x, ids, skip = self._conv_block_4(x)
        fea_list.append(skip)
        ids_list.append(ids)
        x, ids, skip = self._conv_block_5(x)
        fea_list.append(skip)
        ids_list.append(ids)
        x = F.relu(self._conv_6(x))
        fea_list.append(x)
        return fea_list

    def init_weight(self):
        if self.pretrained is not None:
            ppmatting.utils.load_pretrained_model(self, self.pretrained)


@manager.BACKBONES.add_component
def VGG11(**args):
    model = VGGNet(layers=11, **args)
    return model


@manager.BACKBONES.add_component
def VGG13(**args):
    model = VGGNet(layers=13, **args)
    return model


@manager.BACKBONES.add_component
def VGG16(**args):
    model = VGGNet(layers=16, **args)
    return model


@manager.BACKBONES.add_component
def VGG19(**args):
    model = VGGNet(layers=19, **args)
    return model

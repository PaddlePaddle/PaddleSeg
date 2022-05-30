# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
from paddleseg import utils

__all__ = [
    "MobileNetV2_x0_25",
    "MobileNetV2_x0_5",
    "MobileNetV2_x0_75",
    "MobileNetV2_x1_0",
    "MobileNetV2_x1_5",
    "MobileNetV2_x2_0",
]


class MobileNetV2(nn.Layer):
    """
        The MobileNetV2 implementation based on PaddlePaddle.

        The original article refers to
        Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
        "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
        (https://arxiv.org/abs/1801.04381).

        Args:
            scale (float, optional): The scale of channel. Default: 1.0
            pretrained (str, optional): The path or url of pretrained model. Default: None
        """

    def __init__(self, scale=1.0, pretrained=None):
        super().__init__()
        self.scale = scale
        self.pretrained = pretrained
        prefix_name = ""

        bottleneck_params_list = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),  # x4
            (6, 32, 3, 2),  # x8
            (6, 64, 4, 2),
            (6, 96, 3, 1),  # x16
            (6, 160, 3, 2),
            (6, 320, 1, 1),  # x32
        ]
        self.out_index = [1, 2, 4, 6]

        self.conv1 = ConvBNLayer(
            num_channels=3,
            num_filters=int(32 * scale),
            filter_size=3,
            stride=2,
            padding=1,
            name=prefix_name + "conv1_1")

        self.block_list = []
        i = 1
        in_c = int(32 * scale)
        for layer_setting in bottleneck_params_list:
            t, c, n, s = layer_setting
            i += 1
            block = self.add_sublayer(
                prefix_name + "conv" + str(i),
                sublayer=InvresiBlocks(
                    in_c=in_c,
                    t=t,
                    c=int(c * scale),
                    n=n,
                    s=s,
                    name=prefix_name + "conv" + str(i)))
            self.block_list.append(block)
            in_c = int(c * scale)

        out_channels = [
            bottleneck_params_list[idx][1] for idx in self.out_index
        ]
        self.feat_channels = [int(c * scale) for c in out_channels]

        self.init_weight()

    def forward(self, inputs):
        feat_list = []

        y = self.conv1(inputs, if_act=True)
        for idx, block in enumerate(self.block_list):
            y = block(y)
            if idx in self.out_index:
                feat_list.append(y)

        return feat_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 name=None,
                 use_cudnn=True):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            param_attr=ParamAttr(name=name + "_bn_scale"),
            bias_attr=ParamAttr(name=name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, inputs, if_act=True):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if if_act:
            y = F.relu6(y)
        return y


class InvertedResidualUnit(nn.Layer):
    def __init__(self, num_channels, num_in_filter, num_filters, stride,
                 filter_size, padding, expansion_factor, name):
        super(InvertedResidualUnit, self).__init__()
        num_expfilter = int(round(num_in_filter * expansion_factor))
        self._expand_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            name=name + "_expand")

        self._bottleneck_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            num_groups=num_expfilter,
            use_cudnn=False,
            name=name + "_dwise")

        self._linear_conv = ConvBNLayer(
            num_channels=num_expfilter,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            name=name + "_linear")

    def forward(self, inputs, ifshortcut):
        y = self._expand_conv(inputs, if_act=True)
        y = self._bottleneck_conv(y, if_act=True)
        y = self._linear_conv(y, if_act=False)
        if ifshortcut:
            y = paddle.add(inputs, y)
        return y


class InvresiBlocks(nn.Layer):
    def __init__(self, in_c, t, c, n, s, name):
        super(InvresiBlocks, self).__init__()

        self._first_block = InvertedResidualUnit(
            num_channels=in_c,
            num_in_filter=in_c,
            num_filters=c,
            stride=s,
            filter_size=3,
            padding=1,
            expansion_factor=t,
            name=name + "_1")

        self._block_list = []
        for i in range(1, n):
            block = self.add_sublayer(
                name + "_" + str(i + 1),
                sublayer=InvertedResidualUnit(
                    num_channels=c,
                    num_in_filter=c,
                    num_filters=c,
                    stride=1,
                    filter_size=3,
                    padding=1,
                    expansion_factor=t,
                    name=name + "_" + str(i + 1)))
            self._block_list.append(block)

    def forward(self, inputs):
        y = self._first_block(inputs, ifshortcut=False)
        for block in self._block_list:
            y = block(y, ifshortcut=True)
        return y


@manager.BACKBONES.add_component
def MobileNetV2_x0_25(**kwargs):
    model = MobileNetV2(scale=0.25, **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV2_x0_5(**kwargs):
    model = MobileNetV2(scale=0.5, **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV2_x0_75(**kwargs):
    model = MobileNetV2(scale=0.75, **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV2_x1_0(**kwargs):
    model = MobileNetV2(scale=1.0, **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV2_x1_5(**kwargs):
    model = MobileNetV2(scale=1.5, **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV2_x2_0(**kwargs):
    model = MobileNetV2(scale=2.0, **kwargs)
    return model

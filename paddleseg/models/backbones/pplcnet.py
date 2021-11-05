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
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingNormal

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils

__all__ = ["PPLCNet_x0_25", "PPLCNet_x0_5", "PPLCNet_x1_0", "PPLCNet_x1_5"]

NET_CONFIG = {
    #blockname : [k, in_c, out_c, s, use_se]
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}


class PPLCNet(nn.Layer):
    """
    The PPLCNet implementation based on PaddlePaddle.

    The original article refers to Cui, Cheng, Tingquan Gao, Shengyu Wei, Yuning Du, Ruoyu Guo,
    Shuilong Dong, Bin Lu et al. "PP-LCNet: A Lightweight CPU Convolutional Neural Network."
    arXiv preprint arXiv:2109.15099 (2021).

    Args:
        scale (float, optional): The scale of pplcnet. Default: 1.0.
        pretrained (str, optional): The path of pretrained model. Default: None.

    """

    def __init__(self, scale=1.0, pretrained=None):
        super().__init__()
        self.pretrained = pretrained
        self.scale = scale
        self.feat_channels = [
            make_divisible(i * scale) for i in [64, 128, 256, 512]
        ]

        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=2)

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks2"])
        ])

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks4"])
        ])

        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks6"])
        ])

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.blocks2(x)

        x = self.blocks3(x)
        out.append(x)  # 4x

        x = self.blocks4(x)
        out.append(x)  # 8x

        x = self.blocks5(x)
        out.append(x)  # 16x

        x = self.blocks6(x)
        out.append(x)  # 32x

        return out


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 num_groups=1):
        super().__init__()

        self.conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            weight_attr=paddle.ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)

        self.bn = layers.SyncBatchNorm(
            num_filters,
            weight_attr=paddle.ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=paddle.ParamAttr(regularizer=L2Decay(0.0)))
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class DepthwiseSeparable(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dw_size=3,
                 use_se=False):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels)
        if use_se:
            self.se = SEModule(num_channels)
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


@manager.BACKBONES.add_component
def PPLCNet_x0_25(**kwargs):
    """
    PPLCNet_x0_25
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_25` model depends on args.
    """
    model = PPLCNet(scale=0.25, **kwargs)
    return model


@manager.BACKBONES.add_component
def PPLCNet_x0_5(**kwargs):
    """
    PPLCNet_x0_5
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_5` model depends on args.
    """
    model = PPLCNet(scale=0.5, **kwargs)
    return model


@manager.BACKBONES.add_component
def PPLCNet_x1_0(**kwargs):
    """
    PPLCNet_x1_0
    Returns:
        model: nn.Layer. Specific `PPLCNet_x1_0` model depends on args.
    """
    model = PPLCNet(scale=1.0, **kwargs)
    return model


@manager.BACKBONES.add_component
def PPLCNet_x1_5(**kwargs):
    """
    PPLCNet_x1_5
    Returns:
        model: nn.Layer. Specific `PPLCNet_x1_5` model depends on args.
    """
    model = PPLCNet(scale=1.5, **kwargs)
    return model

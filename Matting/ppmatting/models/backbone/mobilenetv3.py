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
import paddle.nn as nn
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn import AdaptiveAvgPool2D, BatchNorm, Conv2D, Dropout, Linear

from paddleseg.cvlibs import manager
from paddleseg.utils import utils, logger
from paddleseg.models import layers

__all__ = [
    "MobileNetV3_small_x0_35", "MobileNetV3_small_x0_5",
    "MobileNetV3_small_x0_75", "MobileNetV3_small_x1_0",
    "MobileNetV3_small_x1_25", "MobileNetV3_large_x0_35",
    "MobileNetV3_large_x0_5", "MobileNetV3_large_x0_75",
    "MobileNetV3_large_x1_0", "MobileNetV3_large_x1_25",
    "MobileNetV3_large_x1_0_os16"
]

MODEL_STAGES_PATTERN = {
    "MobileNetV3_small": ["blocks[0]", "blocks[2]", "blocks[7]", "blocks[10]"],
    "MobileNetV3_large":
    ["blocks[0]", "blocks[2]", "blocks[5]", "blocks[11]", "blocks[14]"]
}

# "large", "small" is just for MobinetV3_large, MobileNetV3_small respectively.
# The type of "large" or "small" config is a list. Each element(list) represents a depthwise block, which is composed of k, exp, se, act, s.
# k: kernel_size
# exp: middle channel number in depthwise block
# c: output channel number in depthwise block
# se: whether to use SE block
# act: which activation to use
# s: stride in depthwise block
# d: dilation rate in depthwise block
NET_CONFIG = {
    "large": [
        # k, exp, c, se, act, s
        [3, 16, 16, False, "relu", 1],
        [3, 64, 24, False, "relu", 2],
        [3, 72, 24, False, "relu", 1],  # x4
        [5, 72, 40, True, "relu", 2],
        [5, 120, 40, True, "relu", 1],
        [5, 120, 40, True, "relu", 1],  # x8
        [3, 240, 80, False, "hardswish", 2],
        [3, 200, 80, False, "hardswish", 1],
        [3, 184, 80, False, "hardswish", 1],
        [3, 184, 80, False, "hardswish", 1],
        [3, 480, 112, True, "hardswish", 1],
        [3, 672, 112, True, "hardswish", 1],  # x16
        [5, 672, 160, True, "hardswish", 2],
        [5, 960, 160, True, "hardswish", 1],
        [5, 960, 160, True, "hardswish", 1],  # x32
    ],
    "small": [
        # k, exp, c, se, act, s
        [3, 16, 16, True, "relu", 2],
        [3, 72, 24, False, "relu", 2],
        [3, 88, 24, False, "relu", 1],
        [5, 96, 40, True, "hardswish", 2],
        [5, 240, 40, True, "hardswish", 1],
        [5, 240, 40, True, "hardswish", 1],
        [5, 120, 48, True, "hardswish", 1],
        [5, 144, 48, True, "hardswish", 1],
        [5, 288, 96, True, "hardswish", 2],
        [5, 576, 96, True, "hardswish", 1],
        [5, 576, 96, True, "hardswish", 1],
    ],
    "large_os8": [
        # k, exp, c, se, act, s, {d}
        [3, 16, 16, False, "relu", 1],
        [3, 64, 24, False, "relu", 2],
        [3, 72, 24, False, "relu", 1],  # x4
        [5, 72, 40, True, "relu", 2],
        [5, 120, 40, True, "relu", 1],
        [5, 120, 40, True, "relu", 1],  # x8
        [3, 240, 80, False, "hardswish", 1],
        [3, 200, 80, False, "hardswish", 1, 2],
        [3, 184, 80, False, "hardswish", 1, 2],
        [3, 184, 80, False, "hardswish", 1, 2],
        [3, 480, 112, True, "hardswish", 1, 2],
        [3, 672, 112, True, "hardswish", 1, 2],
        [5, 672, 160, True, "hardswish", 1, 2],
        [5, 960, 160, True, "hardswish", 1, 4],
        [5, 960, 160, True, "hardswish", 1, 4],
    ],
    "small_os8": [
        # k, exp, c, se, act, s, {d}
        [3, 16, 16, True, "relu", 2],
        [3, 72, 24, False, "relu", 2],
        [3, 88, 24, False, "relu", 1],
        [5, 96, 40, True, "hardswish", 1],
        [5, 240, 40, True, "hardswish", 1, 2],
        [5, 240, 40, True, "hardswish", 1, 2],
        [5, 120, 48, True, "hardswish", 1, 2],
        [5, 144, 48, True, "hardswish", 1, 2],
        [5, 288, 96, True, "hardswish", 1, 2],
        [5, 576, 96, True, "hardswish", 1, 4],
        [5, 576, 96, True, "hardswish", 1, 4],
    ],
    "large_os16": [
        # k, exp, c, se, act, s, {d}
        [3, 16, 16, False, "relu", 1],
        [3, 64, 24, False, "relu", 2],
        [3, 72, 24, False, "relu", 1],  # x4
        [5, 72, 40, True, "relu", 2],
        [5, 120, 40, True, "relu", 1],
        [5, 120, 40, True, "relu", 1],  # x8
        [3, 240, 80, False, "hardswish", 2],
        [3, 200, 80, False, "hardswish", 1, 1],
        [3, 184, 80, False, "hardswish", 1, 1],
        [3, 184, 80, False, "hardswish", 1, 1],
        [3, 480, 112, True, "hardswish", 1, 1],
        [3, 672, 112, True, "hardswish", 1, 1],
        [5, 672, 160, True, "hardswish", 1, 2],
        [5, 960, 160, True, "hardswish", 1, 2],
        [5, 960, 160, True, "hardswish", 1, 2],
    ],
}

OUT_INDEX = {"large": [2, 5, 11, 14], "small": [0, 2, 7, 10]}


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _create_act(act):
    if act == "hardswish":
        return nn.Hardswish()
    elif act == "relu":
        return nn.ReLU()
    elif act is None:
        return None
    else:
        raise RuntimeError(
            "The activation function is not supported: {}".format(act))


class MobileNetV3(nn.Layer):
    """
    MobileNetV3
    Args:
        config: list. MobileNetV3 depthwise blocks config.
        in_channels (int, optional): The channels of input image. Default: 3.
        scale: float=1.0. The coefficient that controls the size of network parameters. 
    Returns:
        model: nn.Layer. Specific MobileNetV3 model depends on args.
    """

    def __init__(self,
                 config,
                 stages_pattern,
                 out_index,
                 in_channels=3,
                 scale=1.0,
                 class_squeeze=960,
                 return_last_conv=False,
                 pretrained=None):
        super().__init__()

        self.cfg = config
        self.out_index = out_index
        self.scale = scale
        self.pretrained = pretrained
        self.class_squeeze = class_squeeze
        self.return_last_conv = return_last_conv
        inplanes = 16

        self.conv = ConvBNLayer(
            in_c=in_channels,
            out_c=_make_divisible(inplanes * self.scale),
            filter_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act="hardswish")
        self.blocks = nn.Sequential(*[
            ResidualUnit(
                in_c=_make_divisible(inplanes * self.scale if i == 0 else
                                     self.cfg[i - 1][2] * self.scale),
                mid_c=_make_divisible(self.scale * exp),
                out_c=_make_divisible(self.scale * c),
                filter_size=k,
                stride=s,
                use_se=se,
                act=act,
                dilation=td[0] if td else 1)
            for i, (k, exp, c, se, act, s, *td) in enumerate(self.cfg)
        ])
        self.last_second_conv = ConvBNLayer(
            in_c=_make_divisible(self.cfg[-1][2] * self.scale),
            out_c=_make_divisible(self.scale * self.class_squeeze),
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act="hardswish")

        # return feat_channels information
        out_channels = [config[idx][2] for idx in out_index]
        if return_last_conv:
            out_channels.append(class_squeeze)
        self.feat_channels = [
            _make_divisible(self.scale * c) for c in out_channels
        ]

        self.mean = paddle.to_tensor([0.485, 0.456, 0.406]).unsqueeze((0, 2, 3))
        self.std = paddle.to_tensor([0.229, 0.224, 0.225]).unsqueeze((0, 2, 3))

        self.init_res(stages_pattern)
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def init_res(self, stages_pattern, return_patterns=None,
                 return_stages=None):
        if return_patterns and return_stages:
            msg = f"The 'return_patterns' would be ignored when 'return_stages' is set."
            logger.warning(msg)
            return_stages = None

        if return_stages is True:
            return_patterns = stages_pattern
        # return_stages is int or bool
        if type(return_stages) is int:
            return_stages = [return_stages]
        if isinstance(return_stages, list):
            if max(return_stages) > len(stages_pattern) or min(
                    return_stages) < 0:
                msg = f"The 'return_stages' set error. Illegal value(s) have been ignored. The stages' pattern list is {stages_pattern}."
                logger.warning(msg)
                return_stages = [
                    val for val in return_stages
                    if val >= 0 and val < len(stages_pattern)
                ]
            return_patterns = [stages_pattern[i] for i in return_stages]

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.conv(x)

        feat_list = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.out_index:
                feat_list.append(x)
        x = self.last_second_conv(x)
        if self.return_last_conv:
            feat_list.append(x)

        return feat_list


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 if_act=True,
                 act=None,
                 dilation=1):
        super().__init__()

        self.conv = Conv2D(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False,
            dilation=dilation)
        self.bn = BatchNorm(
            num_channels=out_c,
            act=None,
            epsilon=0.001,
            momentum=0.99,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.if_act = if_act
        self.act = _create_act(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x


class ResidualUnit(nn.Layer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 act=None,
                 dilation=1):
        super().__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se
        self.in_c = in_c
        self.mid_c = mid_c

        # There is not expand conv in pytorch version when in_c equaled to mid_c.

        if in_c != mid_c:
            self.expand_conv = ConvBNLayer(
                in_c=in_c,
                out_c=mid_c,
                filter_size=1,
                stride=1,
                padding=0,
                if_act=True,
                act=act)
        self.bottleneck_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2) * dilation,
            num_groups=mid_c,
            if_act=True,
            act=act,
            dilation=dilation)
        if self.if_se:
            self.mid_se = SEModule(mid_c)
        self.linear_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    def forward(self, x):
        identity = x
        if self.in_c != self.mid_c:
            x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.add(identity, x)
        return x


# nn.Hardsigmoid can't transfer "slope" and "offset" in nn.functional.hardsigmoid
class Hardsigmoid(nn.Layer):
    def __init__(self, slope=0.2, offset=0.5):
        super().__init__()
        self.slope = slope
        self.offset = offset

    def forward(self, x):
        return nn.functional.hardsigmoid(
            x, slope=self.slope, offset=self.offset)


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=_make_divisible(channel // reduction, 8),
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = Conv2D(
            in_channels=_make_divisible(channel // reduction, 8),
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = Hardsigmoid(slope=0.1666667, offset=0.5)

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return paddle.multiply(x=identity, y=x)


@manager.BACKBONES.add_component
def MobileNetV3_small_x0_35(**kwargs):
    model = MobileNetV3(
        config=NET_CONFIG["small"],
        scale=0.35,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        out_index=OUT_INDEX["small"],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV3_small_x0_5(**kwargs):
    model = MobileNetV3(
        config=NET_CONFIG["small"],
        scale=0.5,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        out_index=OUT_INDEX["small"],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV3_small_x0_75(**kwargs):
    model = MobileNetV3(
        config=NET_CONFIG["small"],
        scale=0.75,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        out_index=OUT_INDEX["small"],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV3_small_x1_0(**kwargs):
    model = MobileNetV3(
        config=NET_CONFIG["small"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        out_index=OUT_INDEX["small"],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV3_small_x1_25(**kwargs):
    model = MobileNetV3(
        config=NET_CONFIG["small"],
        scale=1.25,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        out_index=OUT_INDEX["small"],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV3_large_x0_35(**kwargs):
    model = MobileNetV3(
        config=NET_CONFIG["large"],
        scale=0.35,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        out_index=OUT_INDEX["large"],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV3_large_x0_5(**kwargs):
    model = MobileNetV3(
        config=NET_CONFIG["large"],
        scale=0.5,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        out_index=OUT_INDEX["large"],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV3_large_x0_75(**kwargs):
    model = MobileNetV3(
        config=NET_CONFIG["large"],
        scale=0.75,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        out_index=OUT_INDEX["large"],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV3_large_x1_0(**kwargs):
    model = MobileNetV3(
        config=NET_CONFIG["large"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        out_index=OUT_INDEX["large"],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV3_large_x1_25(**kwargs):
    model = MobileNetV3(
        config=NET_CONFIG["large"],
        scale=1.25,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        out_index=OUT_INDEX["large"],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV3_large_x1_0_os8(**kwargs):
    model = MobileNetV3(
        config=NET_CONFIG["large_os8"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
        out_index=OUT_INDEX["large"],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV3_small_x1_0_os8(**kwargs):
    model = MobileNetV3(
        config=NET_CONFIG["small_os8"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_small"],
        out_index=OUT_INDEX["small"],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV3_large_x1_0_os16(**kwargs):
    if 'out_index' in kwargs:
        model = MobileNetV3(
            config=NET_CONFIG["large_os16"],
            scale=1.0,
            stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
            **kwargs)
    else:
        model = MobileNetV3(
            config=NET_CONFIG["large_os16"],
            scale=1.0,
            stages_pattern=MODEL_STAGES_PATTERN["MobileNetV3_large"],
            out_index=OUT_INDEX["large"],
            **kwargs)
    return model

# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import, division, print_function
import math
import paddle
from paddle import ParamAttr, reshape, transpose, concat, split
import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay

from paddleclas.ppcls.arch.backbone.base.theseus_layer import TheseusLayer
from paddleclas.ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "ESNet_x0_25":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_25_pretrained.pdparams",
    "ESNet_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_5_pretrained.pdparams",
    "ESNet_x0_75":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_75_pretrained.pdparams",
    "ESNet_x1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x1_0_pretrained.pdparams",
}

MODEL_STAGES_PATTERN = {"ESNet": ["blocks[2]", "blocks[9]", "blocks[12]"]}

__all__ = list(MODEL_URLS.keys())


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.shape[0:4]
    channels_per_group = num_channels // groups
    x = reshape(
        x=x, shape=[batch_size, groups, channels_per_group, height, width])
    x = transpose(x=x, perm=[0, 2, 1, 3, 4])
    x = reshape(x=x, shape=[batch_size, num_channels, height, width])
    return x


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 if_act=True):
        super().__init__()
        self.conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)

        self.bn = BatchNorm(
            out_channels,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.if_act = if_act
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.hardswish(x)
        return x


class SEModule(TheseusLayer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = Conv2D(
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


class ESBlock1(TheseusLayer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pw_1_1 = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1)
        self.dw_1 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=1,
            groups=out_channels // 2,
            if_act=False)
        self.se = SEModule(out_channels)

        self.pw_1_2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1)

    def forward(self, x):
        x1, x2 = split(
            x, num_or_sections=[x.shape[1] // 2, x.shape[1] // 2], axis=1)
        x2 = self.pw_1_1(x2)
        x3 = self.dw_1(x2)
        x3 = concat([x2, x3], axis=1)
        x3 = self.se(x3)
        x3 = self.pw_1_2(x3)
        x = concat([x1, x3], axis=1)
        return channel_shuffle(x, 2)


class ESBlock2(TheseusLayer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # branch1
        self.dw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            groups=in_channels,
            if_act=False)
        self.pw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1)
        # branch2
        self.pw_2_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1)
        self.dw_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=2,
            groups=out_channels // 2,
            if_act=False)
        self.se = SEModule(out_channels // 2)
        self.pw_2_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1)
        self.concat_dw = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            groups=out_channels)
        self.concat_pw = ConvBNLayer(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.dw_1(x)
        x1 = self.pw_1(x1)
        x2 = self.pw_2_1(x)
        x2 = self.dw_2(x2)
        x2 = self.se(x2)
        x2 = self.pw_2_2(x2)
        x = concat([x1, x2], axis=1)
        x = self.concat_dw(x)
        x = self.concat_pw(x)
        return x


class ESNet(TheseusLayer):
    def __init__(self,
                 stages_pattern,
                 class_num=1000,
                 scale=1.0,
                 dropout_prob=0.2,
                 class_expand=1280,
                 return_patterns=None,
                 return_stages=None):
        super().__init__()
        self.scale = scale
        self.class_num = class_num
        self.class_expand = class_expand
        stage_repeats = [3, 7, 3]
        stage_out_channels = [
            -1, 24, make_divisible(116 * scale), make_divisible(232 * scale),
            make_divisible(464 * scale), 1024
        ]

        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2)
        self.max_pool = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.LayerList()
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                if i == 0:
                    block = ESBlock2(
                        in_channels=stage_out_channels[stage_id + 1],
                        out_channels=stage_out_channels[stage_id + 2])
                else:
                    block = ESBlock1(
                        in_channels=stage_out_channels[stage_id + 2],
                        out_channels=stage_out_channels[stage_id + 2])
                self.blocks.append(block)

        # self.conv2 = ConvBNLayer(
        #     in_channels=stage_out_channels[-2],
        #     out_channels=stage_out_channels[-1],
        #     kernel_size=1)

        # self.avg_pool = AdaptiveAvgPool2D(1)

        # self.last_conv = Conv2D(
        #     in_channels=stage_out_channels[-1],
        #     out_channels=self.class_expand,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     bias_attr=False)
        # self.hardswish = nn.Hardswish()
        # self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")
        # self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        # self.fc = Linear(self.class_expand, self.class_num)

        super().init_res(
            stages_pattern,
            return_patterns=return_patterns,
            return_stages=return_stages)

    def forward(self, x):  # [4, 3, 512, 512]
        output_indices = [2, 9, 12]
        x = self.conv1(x)  #[4, 24, 256, 256]
        x = self.max_pool(x)  # x4
        outputs = [x]
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in output_indices:
                outputs.append(x)

        # x = self.conv2(x)
        # x = self.avg_pool(x)
        # x = self.last_conv(x)
        # x = self.hardswish(x)
        # x = self.dropout(x)
        # x = self.flatten(x)
        # x = self.fc(x)
        return outputs


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def ESNet_x0_25(pretrained=False, use_ssld=False, **kwargs):
    """
    ESNet_x0_25
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ESNet_x0_25` model depends on args.
    """
    model = ESNet(
        scale=0.25, stages_pattern=MODEL_STAGES_PATTERN["ESNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ESNet_x0_25"], use_ssld)
    return model


def ESNet_x0_5(pretrained=False, use_ssld=False, **kwargs):
    """
    ESNet_x0_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ESNet_x0_5` model depends on args.
    """
    model = ESNet(
        scale=0.5, stages_pattern=MODEL_STAGES_PATTERN["ESNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ESNet_x0_5"], use_ssld)
    return model


def ESNet_x0_75(pretrained=False, use_ssld=False, **kwargs):
    """
    ESNet_x0_75
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ESNet_x0_75` model depends on args.
    """
    model = ESNet(
        scale=0.75, stages_pattern=MODEL_STAGES_PATTERN["ESNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ESNet_x0_75"], use_ssld)
    return model


def ESNet_x1_0(pretrained=False, use_ssld=False, **kwargs):
    """
    ESNet_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ESNet_x1_0` model depends on args.
    """
    model = ESNet(
        scale=1.0, stages_pattern=MODEL_STAGES_PATTERN["ESNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ESNet_x1_0"], use_ssld)
    return model

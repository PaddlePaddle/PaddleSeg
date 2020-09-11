# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, Dropout
from paddle.nn import SyncBatchNorm as BatchNorm

from paddleseg.models.common import layer_utils
from paddleseg.cvlibs import manager
from paddleseg.utils import utils

__all__ = [
    "MobileNetV3_small_x0_35", "MobileNetV3_small_x0_5",
    "MobileNetV3_small_x0_75", "MobileNetV3_small_x1_0",
    "MobileNetV3_small_x1_25", "MobileNetV3_large_x0_35",
    "MobileNetV3_large_x0_5", "MobileNetV3_large_x0_75",
    "MobileNetV3_large_x1_0", "MobileNetV3_large_x1_25"
]


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_padding_same(kernel_size, dilation_rate):
    """
    SAME padding implementation given kernel_size and dilation_rate.
    The calculation formula as following:
        (F-(k+(k -1)*(r-1))+2*p)/s + 1 = F_new
        where F: a feature map
              k: kernel size, r: dilation rate, p: padding value, s: stride
              F_new: new feature map
    Args:
        kernel_size (int)
        dilation_rate (int)

    Returns:
        padding_same (int): padding value
    """
    k = kernel_size
    r = dilation_rate
    padding_same = (k + (k - 1) * (r - 1) - 1) // 2

    return padding_same


class MobileNetV3(fluid.dygraph.Layer):
    def __init__(self,
                 backbone_pretrained=None,
                 scale=1.0,
                 model_name="small",
                 class_dim=1000,
                 output_stride=None):
        super(MobileNetV3, self).__init__()

        inplanes = 16
        if model_name == "large":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, "relu", 1],
                [3, 64, 24, False, "relu", 2],
                [3, 72, 24, False, "relu", 1],  # output 1 -> out_index=2
                [5, 72, 40, True, "relu", 2],
                [5, 120, 40, True, "relu", 1],
                [5, 120, 40, True, "relu", 1],  # output 2 -> out_index=5
                [3, 240, 80, False, "hard_swish", 2],
                [3, 200, 80, False, "hard_swish", 1],
                [3, 184, 80, False, "hard_swish", 1],
                [3, 184, 80, False, "hard_swish", 1],
                [3, 480, 112, True, "hard_swish", 1],
                [3, 672, 112, True, "hard_swish",
                 1],  # output 3 -> out_index=11
                [5, 672, 160, True, "hard_swish", 2],
                [5, 960, 160, True, "hard_swish", 1],
                [5, 960, 160, True, "hard_swish",
                 1],  # output 3 -> out_index=14
            ]
            self.out_indices = [2, 5, 11, 14]

            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        elif model_name == "small":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, "relu", 2],  # output 1 -> out_index=0
                [3, 72, 24, False, "relu", 2],
                [3, 88, 24, False, "relu", 1],  # output 2 -> out_index=3
                [5, 96, 40, True, "hard_swish", 2],
                [5, 240, 40, True, "hard_swish", 1],
                [5, 240, 40, True, "hard_swish", 1],
                [5, 120, 48, True, "hard_swish", 1],
                [5, 144, 48, True, "hard_swish", 1],  # output 3 -> out_index=7
                [5, 288, 96, True, "hard_swish", 2],
                [5, 576, 96, True, "hard_swish", 1],
                [5, 576, 96, True, "hard_swish", 1],  # output 4 -> out_index=10
            ]
            self.out_indices = [0, 3, 7, 10]

            self.cls_ch_squeeze = 576
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError(
                "mode[{}_model] is not implemented!".format(model_name))

        ###################################################
        # modify stride and dilation based on output_stride
        self.dilation_cfg = [1] * len(self.cfg)
        self.modify_bottle_params(output_stride=output_stride)
        ###################################################

        self.conv1 = ConvBNLayer(
            in_c=3,
            out_c=make_divisible(inplanes * scale),
            filter_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act="hard_swish",
            name="conv1")

        self.block_list = []

        inplanes = make_divisible(inplanes * scale)
        for i, (k, exp, c, se, nl, s) in enumerate(self.cfg):
            ######################################
            # add dilation rate
            dilation_rate = self.dilation_cfg[i]
            ######################################
            self.block_list.append(
                ResidualUnit(
                    in_c=inplanes,
                    mid_c=make_divisible(scale * exp),
                    out_c=make_divisible(scale * c),
                    filter_size=k,
                    stride=s,
                    dilation=dilation_rate,
                    use_se=se,
                    act=nl,
                    name="conv" + str(i + 2)))
            self.add_sublayer(
                sublayer=self.block_list[-1], name="conv" + str(i + 2))
            inplanes = make_divisible(scale * c)

        self.last_second_conv = ConvBNLayer(
            in_c=inplanes,
            out_c=make_divisible(scale * self.cls_ch_squeeze),
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act="hard_swish",
            name="conv_last")

        self.pool = Pool2D(
            pool_type="avg", global_pooling=True, use_cudnn=False)

        self.last_conv = Conv2D(
            num_channels=make_divisible(scale * self.cls_ch_squeeze),
            num_filters=self.cls_ch_expand,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(name="last_1x1_conv_weights"),
            bias_attr=False)

        self.out = Linear(
            input_dim=self.cls_ch_expand,
            output_dim=class_dim,
            param_attr=ParamAttr("fc_weights"),
            bias_attr=ParamAttr(name="fc_offset"))

        self.init_weight(backbone_pretrained)

    def modify_bottle_params(self, output_stride=None):

        if output_stride is not None and output_stride % 2 != 0:
            raise Exception("output stride must to be even number")
        if output_stride is not None:
            stride = 2
            rate = 1
            for i, _cfg in enumerate(self.cfg):
                stride = stride * _cfg[-1]
                if stride > output_stride:
                    rate = rate * _cfg[-1]
                    self.cfg[i][-1] = 1

                self.dilation_cfg[i] = rate

    def forward(self, inputs, label=None, dropout_prob=0.2):
        x = self.conv1(inputs)
        # A feature list saves each downsampling feature.
        feat_list = []
        for i, block in enumerate(self.block_list):
            x = block(x)
            if i in self.out_indices:
                feat_list.append(x)
            #print("block {}:".format(i),x.shape, self.dilation_cfg[i])
        x = self.last_second_conv(x)
        x = self.pool(x)
        x = self.last_conv(x)
        x = fluid.layers.hard_swish(x)
        x = fluid.layers.dropout(x=x, dropout_prob=dropout_prob)
        x = fluid.layers.reshape(x, shape=[x.shape[0], x.shape[1]])
        x = self.out(x)

        return x, feat_list

    def init_weight(self, pretrained_model=None):
        """
        Initialize the parameters of model parts.
        Args:
            pretrained_model ([str], optional): the path of pretrained model. Defaults to None.
        """
        if pretrained_model is not None:
            if os.path.exists(pretrained_model):
                utils.load_pretrained_model(self, pretrained_model)
            else:
                raise Exception('Pretrained model is not found: {}'.format(
                    pretrained_model))


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 dilation=1,
                 num_groups=1,
                 if_act=True,
                 act=None,
                 use_cudnn=True,
                 name=""):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act

        self.conv = fluid.dygraph.Conv2D(
            num_channels=in_c,
            num_filters=out_c,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=num_groups,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            use_cudnn=use_cudnn,
            act=None)
        self.bn = BatchNorm(
            num_features=out_c,
            weight_attr=ParamAttr(
                name=name + "_bn_scale",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            bias_attr=ParamAttr(
                name=name + "_bn_offset",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)))

        self._act_op = layer_utils.Activation(act=None)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = fluid.layers.relu(x)
            elif self.act == "hard_swish":
                x = fluid.layers.hard_swish(x)
            else:
                print("The activation function is selected incorrectly.")
                exit()
        return x


class ResidualUnit(fluid.dygraph.Layer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 dilation=1,
                 act=None,
                 name=''):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + "_expand")

        self.bottleneck_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=get_padding_same(
                filter_size,
                dilation),  #int((filter_size - 1) // 2) + (dilation - 1),
            dilation=dilation,
            num_groups=mid_c,
            if_act=True,
            act=act,
            name=name + "_depthwise")
        if self.if_se:
            self.mid_se = SEModule(mid_c, name=name + "_se")
        self.linear_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name=name + "_linear")
        self.dilation = dilation

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = fluid.layers.elementwise_add(inputs, x)
        return x


class SEModule(fluid.dygraph.Layer):
    def __init__(self, channel, reduction=4, name=""):
        super(SEModule, self).__init__()
        self.avg_pool = fluid.dygraph.Pool2D(
            pool_type="avg", global_pooling=True, use_cudnn=False)
        self.conv1 = fluid.dygraph.Conv2D(
            num_channels=channel,
            num_filters=channel // reduction,
            filter_size=1,
            stride=1,
            padding=0,
            act="relu",
            param_attr=ParamAttr(name=name + "_1_weights"),
            bias_attr=ParamAttr(name=name + "_1_offset"))
        self.conv2 = fluid.dygraph.Conv2D(
            num_channels=channel // reduction,
            num_filters=channel,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(name + "_2_weights"),
            bias_attr=ParamAttr(name=name + "_2_offset"))

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = fluid.layers.hard_sigmoid(outputs)
        return fluid.layers.elementwise_mul(x=inputs, y=outputs, axis=0)


def MobileNetV3_small_x0_35(**kwargs):
    model = MobileNetV3(model_name="small", scale=0.35, **kwargs)
    return model


def MobileNetV3_small_x0_5(**kwargs):
    model = MobileNetV3(model_name="small", scale=0.5, **kwargs)
    return model


def MobileNetV3_small_x0_75(**kwargs):
    model = MobileNetV3(model_name="small", scale=0.75, **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV3_small_x1_0(**kwargs):
    model = MobileNetV3(model_name="small", scale=1.0, **kwargs)
    return model


def MobileNetV3_small_x1_25(**kwargs):
    model = MobileNetV3(model_name="small", scale=1.25, **kwargs)
    return model


def MobileNetV3_large_x0_35(**kwargs):
    model = MobileNetV3(model_name="large", scale=0.35, **kwargs)
    return model


def MobileNetV3_large_x0_5(**kwargs):
    model = MobileNetV3(model_name="large", scale=0.5, **kwargs)
    return model


def MobileNetV3_large_x0_75(**kwargs):
    model = MobileNetV3(model_name="large", scale=0.75, **kwargs)
    return model


@manager.BACKBONES.add_component
def MobileNetV3_large_x1_0(**kwargs):
    model = MobileNetV3(model_name="large", scale=1.0, **kwargs)
    return model


def MobileNetV3_large_x1_25(**kwargs):
    model = MobileNetV3(model_name="large", scale=1.25, **kwargs)
    return model

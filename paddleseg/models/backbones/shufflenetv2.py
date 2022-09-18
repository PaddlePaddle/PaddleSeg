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
from paddle import ParamAttr, reshape, transpose, concat, split
from paddle.nn import Layer, Conv2D, MaxPool2D, AdaptiveAvgPool2D, BatchNorm, Linear
from paddle.nn.initializer import KaimingNormal
from paddle.nn.functional import swish

from paddleseg.cvlibs import manager
from paddleseg.utils import utils, logger

__all__ = [
    'ShuffleNetV2_x0_25', 'ShuffleNetV2_x0_33', 'ShuffleNetV2_x0_5',
    'ShuffleNetV2_x1_0', 'ShuffleNetV2_x1_5', 'ShuffleNetV2_x2_0',
    'ShuffleNetV2_swish'
]


def channel_shuffle(x, groups):
    x_shape = paddle.shape(x)
    batch_size, height, width = x_shape[0], x_shape[2], x_shape[3]
    num_channels = x.shape[1]
    channels_per_group = num_channels // groups

    # reshape
    x = reshape(
        x=x, shape=[batch_size, groups, channels_per_group, height, width])

    # transpose
    x = transpose(x=x, perm=[0, 2, 1, 3, 4])

    # flatten
    x = reshape(x=x, shape=[batch_size, num_channels, height, width])

    return x


class ConvBNLayer(Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=1,
            act=None,
            name=None, ):
        super(ConvBNLayer, self).__init__()
        self._conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(
                initializer=KaimingNormal(), name=name + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            out_channels,
            param_attr=ParamAttr(name=name + "_bn_scale"),
            bias_attr=ParamAttr(name=name + "_bn_offset"),
            act=act,
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class InvertedResidual(Layer):
    def __init__(self, in_channels, out_channels, stride, act="relu",
                 name=None):
        super(InvertedResidual, self).__init__()
        self._conv_pw = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv1')
        self._conv_dw = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels // 2,
            act=None,
            name='stage_' + name + '_conv2')
        self._conv_linear = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv3')

    def forward(self, inputs):
        x1, x2 = split(
            inputs,
            num_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
            axis=1)
        x2 = self._conv_pw(x2)
        x2 = self._conv_dw(x2)
        x2 = self._conv_linear(x2)
        out = concat([x1, x2], axis=1)
        return channel_shuffle(out, 2)


class InvertedResidualDS(Layer):
    def __init__(self, in_channels, out_channels, stride, act="relu",
                 name=None):
        super(InvertedResidualDS, self).__init__()

        # branch1
        self._conv_dw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            act=None,
            name='stage_' + name + '_conv4')
        self._conv_linear_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv5')
        # branch2
        self._conv_pw_2 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv1')
        self._conv_dw_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels // 2,
            act=None,
            name='stage_' + name + '_conv2')
        self._conv_linear_2 = ConvBNLayer(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act,
            name='stage_' + name + '_conv3')

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._conv_linear_2(x2)
        out = concat([x1, x2], axis=1)

        return channel_shuffle(out, 2)


class ShuffleNet(Layer):
    def __init__(self, scale=1.0, act="relu", in_channels=3, pretrained=None):
        super(ShuffleNet, self).__init__()
        self.scale = scale
        self.pretrained = pretrained
        stage_repeats = [4, 8, 4]

        if scale == 0.25:
            stage_out_channels = [-1, 24, 24, 48, 96, 512]
        elif scale == 0.33:
            stage_out_channels = [-1, 24, 32, 64, 128, 512]
        elif scale == 0.5:
            stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif scale == 1.0:
            stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif scale == 1.5:
            stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif scale == 2.0:
            stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise NotImplementedError("This scale size:[" + str(scale) +
                                      "] is not implemented!")

        self.out_index = [3, 11, 15]
        self.feat_channels = stage_out_channels[1:5]

        # 1. conv1
        self._conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            act=act,
            name='stage1_conv')
        self._max_pool = MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 2. bottleneck sequences
        self._block_list = []
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                if i == 0:
                    block = self.add_sublayer(
                        name=str(stage_id + 2) + '_' + str(i + 1),
                        sublayer=InvertedResidualDS(
                            in_channels=stage_out_channels[stage_id + 1],
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=2,
                            act=act,
                            name=str(stage_id + 2) + '_' + str(i + 1)))
                else:
                    block = self.add_sublayer(
                        name=str(stage_id + 2) + '_' + str(i + 1),
                        sublayer=InvertedResidual(
                            in_channels=stage_out_channels[stage_id + 2],
                            out_channels=stage_out_channels[stage_id + 2],
                            stride=1,
                            act=act,
                            name=str(stage_id + 2) + '_' + str(i + 1)))
                self._block_list.append(block)

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, inputs):
        feat_list = []

        y = self._conv1(inputs)
        y = self._max_pool(y)
        feat_list.append(y)

        for idx, inv in enumerate(self._block_list):
            y = inv(y)
            if idx in self.out_index:
                feat_list.append(y)
        return feat_list


@manager.BACKBONES.add_component
def ShuffleNetV2_x0_25(**kwargs):
    model = ShuffleNet(scale=0.25, **kwargs)
    return model


@manager.BACKBONES.add_component
def ShuffleNetV2_x0_33(**kwargs):
    model = ShuffleNet(scale=0.33, **kwargs)
    return model


@manager.BACKBONES.add_component
def ShuffleNetV2_x0_5(**kwargs):
    model = ShuffleNet(scale=0.5, **kwargs)
    return model


@manager.BACKBONES.add_component
def ShuffleNetV2_x1_0(**kwargs):
    model = ShuffleNet(scale=1.0, **kwargs)
    return model


@manager.BACKBONES.add_component
def ShuffleNetV2_x1_5(**kwargs):
    model = ShuffleNet(scale=1.5, **kwargs)
    return model


@manager.BACKBONES.add_component
def ShuffleNetV2_x2_0(**kwargs):
    model = ShuffleNet(scale=2.0, **kwargs)
    return model


@manager.BACKBONES.add_component
def ShuffleNetV2_swish(**kwargs):
    model = ShuffleNet(scale=1.0, act="swish", **kwargs)
    return model

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

from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.utils import utils

__all__ = ['ShuffleNetV2']


@manager.MODELS.add_component
class ShuffleNetV2(nn.Layer):
    def __init__(self, num_classes, pretrained=None, align_corners=False):
        super().__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.align_corners = align_corners

        self.conv_bn0 = _ConvBNReLU(3, 36, 3, 2, 1)
        self.conv_bn1 = _ConvBNReLU(36, 18, 1, 1, 0)

        self.block1 = nn.Sequential(
            SFNetV2Module(36, stride=2, out_channels=72),
            SFNetV2Module(72, stride=1), SFNetV2Module(72, stride=1),
            SFNetV2Module(72, stride=1))

        self.block2 = nn.Sequential(
            SFNetV2Module(72, stride=2), SFNetV2Module(144, stride=1),
            SFNetV2Module(144, stride=1), SFNetV2Module(144, stride=1),
            SFNetV2Module(144, stride=1), SFNetV2Module(144, stride=1),
            SFNetV2Module(144, stride=1), SFNetV2Module(144, stride=1))

        self.depthwise_separable0 = _SeparableConvBNReLU(144, 64, 3, stride=1)
        self.depthwise_separable1 = _SeparableConvBNReLU(82, 64, 3, stride=1)

        weight_attr = paddle.ParamAttr(
            learning_rate=1.,
            regularizer=paddle.regularizer.L2Decay(coeff=0.),
            initializer=nn.initializer.XavierUniform())
        self.deconv = nn.Conv2DTranspose(
            64,
            self.num_classes,
            2,
            stride=2,
            padding=0,
            weight_attr=weight_attr,
            bias_attr=True)

        self.init_weight()

    def forward(self, x):
        ## Encoder
        conv1 = self.conv_bn0(x)  # encoder 1
        shortcut = self.conv_bn1(conv1)  # shortcut 1

        pool = F.max_pool2d(
            conv1, kernel_size=3, stride=2, padding=1)  # encoder 2

        # Block 1
        conv = self.block1(pool)  # encoder 3

        # Block 2
        conv = self.block2(conv)  # encoder 4

        ### decoder
        conv = self.depthwise_separable0(conv)
        shortcut_shape = paddle.shape(shortcut)[2:]
        conv_b = F.interpolate(
            conv,
            shortcut_shape,
            mode='bilinear',
            align_corners=self.align_corners)
        concat = paddle.concat(x=[shortcut, conv_b], axis=1)
        decode_conv = self.depthwise_separable1(concat)
        logit = self.deconv(decode_conv)
        return [logit]

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)


class _ConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 **kwargs):
        super().__init__()
        weight_attr = paddle.ParamAttr(
            learning_rate=1, initializer=nn.initializer.KaimingUniform())
        self._conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=False,
            **kwargs)

        self._batch_norm = layers.SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x


class _ConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 **kwargs):
        super().__init__()
        weight_attr = paddle.ParamAttr(
            learning_rate=1, initializer=nn.initializer.KaimingUniform())
        self._conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=False,
            **kwargs)

        self._batch_norm = layers.SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class _SeparableConvBNReLU(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise_conv = _ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size / 2),
            groups=in_channels,
            **kwargs)
        self.piontwise_conv = _ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=1,
            stride=1,
            padding=0)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class SFNetV2Module(nn.Layer):
    def __init__(self, input_channels, stride, out_channels=None):
        super().__init__()
        if stride == 1:
            branch_channel = int(input_channels / 2)
        else:
            branch_channel = input_channels

        if out_channels is None:
            self.in_channels = int(branch_channel)
        else:
            self.in_channels = int(out_channels / 2)

        self._depthwise_separable_0 = _SeparableConvBNReLU(
            input_channels, self.in_channels, 3, stride=stride)
        self._conv = _ConvBNReLU(
            branch_channel, self.in_channels, 1, stride=1, padding=0)
        self._depthwise_separable_1 = _SeparableConvBNReLU(
            self.in_channels, self.in_channels, 3, stride=stride)

        self.stride = stride

    def forward(self, input):

        if self.stride == 1:
            shortcut, branch = paddle.split(x=input, num_or_sections=2, axis=1)
        else:
            branch = input
            shortcut = self._depthwise_separable_0(input)

        branch_1x1 = self._conv(branch)
        branch_dw1x1 = self._depthwise_separable_1(branch_1x1)
        output = paddle.concat(x=[shortcut, branch_dw1x1], axis=1)

        # channel shuffle
        out_shape = paddle.shape(output)
        h, w = out_shape[2], out_shape[3]
        output = paddle.reshape(x=output, shape=[0, 2, self.in_channels, h, w])
        output = paddle.transpose(x=output, perm=[0, 2, 1, 3, 4])
        output = paddle.reshape(x=output, shape=[0, 2 * self.in_channels, h, w])
        return output


if __name__ == '__main__':
    import numpy as np
    import os

    np.random.seed(100)
    paddle.seed(100)

    net = ShuffleNetV2(10)
    img = np.random.random(size=(4, 3, 100, 100)).astype('float32')
    img = paddle.to_tensor(img)
    out = net(img)
    print(out)

    net.forward = paddle.jit.to_static(net.forward)
    save_path = os.path.join('.', 'model')
    in_var = paddle.ones([4, 3, 100, 100])
    paddle.jit.save(net, save_path, input_spec=[in_var])

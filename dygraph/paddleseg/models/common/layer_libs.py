# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import nn
import paddle.nn.functional as F
from paddle.nn import Conv2d
from paddle.nn import SyncBatchNorm as BatchNorm


class ConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):

        super(ConvBNReLU, self).__init__()

        self._conv = Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        self._batch_norm = BatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x


class ConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):

        super(ConvBN, self).__init__()
        self._conv = Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        self._batch_norm = BatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvReluPool(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(ConvReluPool, self).__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = F.pool2d(x, pool_size=2, pool_type="max", pool_stride=2)
        return x


class SeparableConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super(SeparableConvBNReLU, self).__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)
        self.piontwise_conv = ConvBNReLU(
            in_channels, out_channels, kernel_size=1, groups=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class DepthwiseConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super(DepthwiseConvBN, self).__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        return x


class AuxLayer(nn.Layer):
    """
    The auxilary layer implementation for auxilary loss

    Args:
        in_channels (int): the number of input channels.
        inter_channels (int): intermediate channels.
        out_channels (int): the number of output channels, which is usually num_classes.
        dropout_prob (float): the droput rate. Default to 0.1.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 dropout_prob=0.1):
        super(AuxLayer, self).__init__()

        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1)

        self.conv = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1)

        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = F.dropout(x, p=self.dropout_prob)
        x = self.conv(x)
        return x

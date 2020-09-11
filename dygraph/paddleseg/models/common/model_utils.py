# -*- encoding: utf-8 -*-
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
from paddle.nn import SyncBatchNorm as BatchNorm

from paddleseg.models.common import layer_utils


class FCNHead(nn.Layer):
    """
    The FCNHead implementation used in auxilary layer

    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__()

        inter_channels = in_channels // 4
        self.conv_bn_relu = layer_utils.ConvBnRelu(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1)

        self.conv = nn.Conv2d(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = F.dropout(x, p=0.1)
        x = self.conv(x)
        return x


class AuxLayer(nn.Layer):
    """
    The auxilary layer implementation for auxilary loss

    Args:
        in_channels (int): the number of input channels.
        inter_channels (int): intermediate channels.
        out_channels (int): the number of output channels, which is usually num_classes.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 dropout_prob=0.1):
        super(AuxLayer, self).__init__()

        self.conv_bn_relu = layer_utils.ConvBnRelu(
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


class PPModule(nn.Layer):
    """
    Pyramid pooling module

    Args:
        in_channels (int): the number of intput channels to pyramid pooling module.

        out_channels (int): the number of output channels after pyramid pooling module.

        bin_sizes (tuple): the out size of pooled feature maps. Default to (1,2,3,6).

        dim_reduction (bool): a bool value represent if reduing dimention after pooling. Default to True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bin_sizes=(1, 2, 3, 6),
                 dim_reduction=True):
        super(PPModule, self).__init__()
        self.bin_sizes = bin_sizes

        inter_channels = in_channels
        if dim_reduction:
            inter_channels = in_channels // len(bin_sizes)

        # we use dimension reduction after pooling mentioned in original implementation.
        self.stages = nn.LayerList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_bn_relu2 = layer_utils.ConvBnRelu(
            in_channels=in_channels + inter_channels * len(bin_sizes),
            out_channels=out_channels,
            kernel_size=3,
            padding=1)

    def _make_stage(self, in_channels, out_channels, size):
        """
        Create one pooling layer.

        In our implementation, we adopt the same dimention reduction as the original paper that might be
        slightly different with other implementations. 

        After pooling, the channels are reduced to 1/len(bin_sizes) immediately, while some other implementations
        keep the channels to be same.


        Args:
            in_channels (int): the number of intput channels to pyramid pooling module.

            size (int): the out size of the pooled layer.

        Returns:
            conv (tensor): a tensor after Pyramid Pooling Module
        """

        # this paddle version does not support AdaptiveAvgPool2d, so skip it here.
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = layer_utils.ConvBnRelu(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        return conv

    def forward(self, input):
        cat_layers = []
        for i, stage in enumerate(self.stages):
            size = self.bin_sizes[i]
            x = F.adaptive_pool2d(
                input, pool_size=(size, size), pool_type="max")
            x = stage(x)
            x = F.resize_bilinear(x, out_shape=input.shape[2:])
            cat_layers.append(x)
        cat_layers = [input] + cat_layers[::-1]
        cat = paddle.concat(cat_layers, axis=1)
        out = self.conv_bn_relu2(cat)

        return out

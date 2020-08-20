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

from paddle import fluid
from paddle.fluid import dygraph
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import SyncBatchNorm as BatchNorm
import cv2
import os
import sys


class ConvBnRelu(dygraph.Layer):

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 using_sep_conv=False,
                 **kwargs):
        
        super(ConvBnRelu, self).__init__()

        if using_sep_conv:
            self.conv = DepthwiseConvBnRelu(num_channels,
                                            num_filters,
                                            filter_size,
                                            **kwargs)
        else:

            self.conv = Conv2D(num_channels,
                                num_filters,
                                filter_size,
                                **kwargs)

        self.batch_norm = BatchNorm(num_filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = fluid.layers.relu(x)
        return x


class ConvBn(dygraph.Layer):
    def __init__(self, num_channels, num_filters, filter_size, **kwargs):
        super(ConvBn, self).__init__()
        self.conv = Conv2D(num_channels,
                           num_filters,
                           filter_size,
                           **kwargs)
        self.batch_norm = BatchNorm(num_filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return x


class ConvReluPool(dygraph.Layer):
    def __init__(self, num_channels, num_filters):
        super(ConvReluPool, self).__init__()
        self.conv = Conv2D(num_channels,
                           num_filters,
                           filter_size=3,
                           stride=1,
                           padding=1,
                           dilation=1)

    def forward(self, x):
        x = self.conv(x)
        x = fluid.layers.relu(x)
        x = fluid.layers.pool2d(x, pool_size=2, pool_type="max", pool_stride=2)
        return x


class ConvBnReluUpsample(dygraph.Layer):
    def __init__(self, num_channels, num_filters):
        super(ConvBnReluUpsample, self).__init__()
        self.conv_bn_relu = ConvBnRelu(num_channels, num_filters)

    def forward(self, x, upsample_scale=2):
        x = self.conv_bn_relu(x)
        new_shape = [x.shape[2] * upsample_scale, x.shape[3] * upsample_scale]
        x = fluid.layers.resize_bilinear(x, new_shape)
        return x


class DepthwiseConvBnRelu(dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 **kwargs):
        super(DepthwiseConvBnRelu, self).__init__()
        self.depthwise_conv = ConvBn(num_channels,
                                    num_filters=num_channels,
                                    filter_size=filter_size,
                                    groups=num_channels,
                                    use_cudnn=False,
                                    **kwargs)
        self.piontwise_conv = ConvBnRelu(num_channels,
                                        num_filters,
                                        filter_size=1,
                                        groups=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


def compute_loss(logits, label, ignore_index=255):
    mask = label != ignore_index
    mask = fluid.layers.cast(mask, 'float32')
    loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits,
        label,
        ignore_index=ignore_index,
        return_softmax=True,
        axis=1)

    loss = loss * mask
    avg_loss = fluid.layers.mean(loss) / (
            fluid.layers.mean(mask) + 1e-5)

    label.stop_gradient = True
    mask.stop_gradient = True
    return avg_loss
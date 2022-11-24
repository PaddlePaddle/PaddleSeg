# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os

import paddle
import paddle.nn as nn


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ConvDropoutNormNonlin(nn.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 conv_op=nn.Conv2D,
                 conv_kwargs=None,
                 norm_op=nn.BatchNorm2D,
                 norm_op_kwargs=None,
                 dropout_op=nn.Dropout2D,
                 dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU,
                 nonlin_kwargs=None):
        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5}
        if conv_kwargs is None:
            conv_kwargs = {
                'kernel_size': 3,
                'stride': 1,
                'padding': 1,
                'dilation': 1,
                'bias_attr': True
            }

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels,
                                 **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs[
                'p'] is not None and self.dropout_op_kwargs['p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)

        x = self.lrelu(self.instnorm(x))
        return x

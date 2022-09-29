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

import paddle
import paddle.nn as nn

from paddleseg.cvlibs import param_init


class PolarizedSelfAttentionModule(nn.Layer):
    '''
    The original article refers to refer to https://arxiv.org/pdf/2107.00782
    
    Args:
        inplanes (int): Input channels of feature.
        planes (int): Output channels of feature.
        kernel_size (int, optional): The kernel size of Conv2D. Default: 1
        stride (int, optional): The stride length of Conv2D. Default: 1
    '''

    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super().__init__()
        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.conv_q_right = nn.Conv2D(
            self.inplanes,
            1,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias_attr=False)
        self.conv_v_right = nn.Conv2D(
            self.inplanes,
            self.inter_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias_attr=False)
        self.conv_up = nn.Conv2D(
            self.inter_planes,
            self.planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False)
        self.softmax_right = nn.Softmax(axis=2)
        self.sigmoid = nn.Sigmoid()
        self.conv_q_left = nn.Conv2D(
            self.inplanes,
            self.inter_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias_attr=False)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv_v_left = nn.Conv2D(
            self.inplanes,
            self.inter_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias_attr=False)
        self.softmax_left = nn.Softmax(axis=2)

        self.init_weight()

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)
        batch, _, height, width = paddle.shape(input_x)
        input_x = input_x.reshape((batch, self.inter_planes, height * width))
        context_mask = self.conv_q_right(x)
        context_mask = context_mask.reshape((batch, 1, height * width))
        context_mask = self.softmax_right(context_mask)
        context = paddle.matmul(input_x, context_mask.transpose((0, 2, 1)))
        context = context.unsqueeze(-1)
        context = self.conv_up(context)
        mask_ch = self.sigmoid(context)
        out = x * mask_ch
        return out

    def channel_pool(self, x):
        g_x = self.conv_q_left(x)
        batch, channel, height, width = paddle.shape(g_x)
        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = paddle.shape(avg_x)
        avg_x = avg_x.reshape((batch, channel, avg_x_h * avg_x_w))
        avg_x = paddle.reshape(avg_x, [batch, avg_x_h * avg_x_w, channel])
        theta_x = self.conv_v_left(x).reshape(
            (batch, self.inter_planes, height * width))
        context = paddle.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.reshape((batch, 1, height, width))
        mask_sp = self.sigmoid(context)
        out = x * mask_sp
        return out

    def forward(self, x):
        context_channel = self.spatial_pool(x)
        context_spatial = self.channel_pool(x)
        out = context_spatial + context_channel
        return out

    def init_weight(self):
        param_init.kaiming_normal_init(self.conv_q_right.weight)
        param_init.kaiming_normal_init(self.conv_v_right.weight)
        param_init.kaiming_normal_init(self.conv_q_left.weight)
        param_init.kaiming_normal_init(self.conv_v_left.weight)
        param_init.kaiming_normal_init(self.conv_up.weight)

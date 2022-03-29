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

from paddleseg.models import layers


class NonLocal2D(nn.Layer):
    """Basic Non-local module.
    This model is the implementation of "Non-local Neural Networks"
    (https://arxiv.org/abs/1711.07971)

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by `1/sqrt(inter_channels)` when the mode is `embedded_gaussian`. Default: True.
        sub_sample (bool): Whether to utilize max pooling after pairwise function. Default: False.
        mode (str): Options are `gaussian`, `concatenation`, `embedded_gaussian` and `dot_product`. Default: embedded_gaussian.
    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 sub_sample=False,
                 mode='embedded_gaussian'):
        super(NonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.sub_sample = sub_sample
        self.mode = mode
        if mode not in [
                'gaussian', 'embedded_gaussian', 'dot_product', 'concatenation'
        ]:
            raise ValueError(
                "Mode should be in 'gaussian', 'concatenation','embedded_gaussian' or 'dot_product'."
            )

        self.inter_channels = max(in_channels // reduction, 1)

        self.g = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1)
        self.conv_out = layers.ConvBNReLU(
            in_channels=self.inter_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            bias_attr=False)

        if self.mode != "gaussian":
            self.theta = nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1)
            self.phi = nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1)

        if self.mode == "concatenation":
            self.concat_project = layers.ConvBNReLU(
                in_channels=self.inter_channels * 2,
                out_channels=1,
                kernel_size=1,
                bias_attr=False)

        if self.sub_sample:
            max_pool_layer = nn.MaxPool2D(kernel_size=(2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer

    def gaussian(self, theta_x, phi_x):
        pairwise_weight = paddle.matmul(theta_x, phi_x)
        pairwise_weight = F.softmax(pairwise_weight, axis=-1)
        return pairwise_weight

    def embedded_gaussian(self, theta_x, phi_x):
        pairwise_weight = paddle.matmul(theta_x, phi_x)
        if self.use_scale:
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = F.softmax(pairwise_weight, -1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        pairwise_weight = paddle.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def concatenation(self, theta_x, phi_x):
        h = theta_x.shape[2]
        w = phi_x.shape[3]
        theta_x = paddle.tile(theta_x, [1, 1, 1, w])
        phi_x = paddle.tile(phi_x, [1, 1, h, 1])

        concat_feature = paddle.concat([theta_x, phi_x], axis=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.shape
        pairwise_weight = paddle.reshape(pairwise_weight, [n, h, w])
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        n, c, h, w = x.shape
        g_x = paddle.reshape(self.g(x), [n, self.inter_channels, -1])
        g_x = paddle.transpose(g_x, [0, 2, 1])

        if self.mode == 'gaussian':
            theta_x = paddle.reshape(x, [n, self.inter_channels, -1])
            theta_x = paddle.transpose(theta_x, [0, 2, 1])
            if self.sub_sample:
                phi_x = paddle.reshape(
                    self.phi(x), [n, self.inter_channels, -1])
            else:
                phi_x = paddle.reshape(x, [n, self.in_channels, -1])

        elif self.mode == 'concatenation':
            theta_x = paddle.reshape(
                self.theta(x), [n, self.inter_channels, -1, 1])
            phi_x = paddle.reshape(self.phi(x), [n, self.inter_channels, 1, -1])

        else:
            theta_x = paddle.reshape(
                self.theta(x), [n, self.inter_channels, -1])
            theta_x = paddle.transpose(theta_x, [0, 2, 1])
            phi_x = paddle.reshape(self.phi(x), [n, self.inter_channels, -1])

        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)
        y = paddle.matmul(pairwise_weight, g_x)
        y = paddle.transpose(y, [0, 2, 1])
        y = paddle.reshape(y, [n, self.inter_channels, h, w])

        output = x + self.conv_out(y)

        return output

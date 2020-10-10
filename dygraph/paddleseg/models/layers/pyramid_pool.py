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
import paddle.nn.functional as F
from paddle import nn

from paddleseg.models import layers


class ASPPModule(nn.Layer):
    """
    Atrous Spatial Pyramid Pooling.

    Args:
        aspp_ratios (tuple): The dilation rate using in ASSP module.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        use_sep_conv (bool, optional): If using separable conv in ASPP module. Default: False.
        image_pooling (bool, optional): If augmented with image-level features. Default: False
    """

    def __init__(self,
                 aspp_ratios,
                 in_channels,
                 out_channels,
                 use_sep_conv=False,
                 image_pooling=False):
        super().__init__()

        self.aspp_blocks = []

        for ratio in aspp_ratios:
            if use_sep_conv and ratio > 1:
                conv_func = layers.SeparableConvBNReLU
            else:
                conv_func = layers.ConvBNReLU

            block = conv_func(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1 if ratio == 1 else 3,
                dilation=ratio,
                padding=0 if ratio == 1 else ratio)
            self.aspp_blocks.append(block)

        out_size = len(self.aspp_blocks)

        if image_pooling:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                layers.ConvBNReLU(
                    in_channels, out_channels, kernel_size=1, bias_attr=False))
            out_size += 1
        self.image_pooling = image_pooling

        self.conv_bn_relu = layers.ConvBNReLU(
            in_channels=out_channels * out_size,
            out_channels=out_channels,
            kernel_size=1)

        self.dropout = nn.Dropout(p=0.1)  # drop rate

    def forward(self, x):
        outputs = []
        for block in self.aspp_blocks:
            y = block(x)
            y = F.resize_bilinear(y, out_shape=x.shape[2:])
            outputs.append(y)

        if self.image_pooling:
            img_avg = self.global_avg_pool(x)
            img_avg = F.resize_bilinear(img_avg, out_shape=x.shape[2:])
            outputs.append(img_avg)

        x = paddle.concat(outputs, axis=1)
        x = self.conv_bn_relu(x)
        x = self.dropout(x)

        return x


class PPModule(nn.Layer):
    """
    Pyramid pooling module originally in PSPNet.

    Args:
        in_channels (int): The number of intput channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 2, 3, 6).
        dim_reduction (bool, optional): A bool value represents if reducing dimension after pooling. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bin_sizes=(1, 2, 3, 6),
                 dim_reduction=True):
        super().__init__()

        self.bin_sizes = bin_sizes

        inter_channels = in_channels
        if dim_reduction:
            inter_channels = in_channels // len(bin_sizes)

        # we use dimension reduction after pooling mentioned in original implementation.
        self.stages = nn.LayerList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_bn_relu2 = layers.ConvBNReLU(
            in_channels=in_channels + inter_channels * len(bin_sizes),
            out_channels=out_channels,
            kernel_size=3,
            padding=1)

    def _make_stage(self, in_channels, out_channels, size):
        """
        Create one pooling layer.

        In our implementation, we adopt the same dimension reduction as the original paper that might be
        slightly different with other implementations.

        After pooling, the channels are reduced to 1/len(bin_sizes) immediately, while some other implementations
        keep the channels to be same.

        Args:
            in_channels (int): The number of intput channels to pyramid pooling module.
            size (int): The out size of the pooled layer.

        Returns:
            conv (tensor): A tensor after Pyramid Pooling Module.
        """

        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = layers.ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        return nn.Sequential(prior, conv)

    def forward(self, input):
        cat_layers = []
        for stage in self.stages:
            x = stage(input)
            x = F.resize_bilinear(x, out_shape=input.shape[2:])
            cat_layers.append(x)
        cat_layers = [input] + cat_layers[::-1]
        cat = paddle.concat(cat_layers, axis=1)
        out = self.conv_bn_relu2(cat)

        return out

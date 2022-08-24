# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/tamasino52/UNETR/blob/main/unetr.py

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

from typing import List, Tuple
import os, sys
sys.path.insert(0, os.getcwd() + '//..//..')
import numpy as np
import paddle
import paddle.nn as nn
from paddle import Tensor

from medicalseg.cvlibs import manager
from medicalseg.models.resnet import ResNet50
from medicalseg.models.vision_transformer import ViT_base_patch16_224


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2D(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True, ):
        conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=not (use_batchnorm), )
        relu = nn.ReLU()

        bn = nn.BatchNorm2D(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True, ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm, )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm, )

        self.up = nn.UpsamplingBilinear2D(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = paddle.concat([x, skip], axis=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Layer):
    def __init__(self, n_skip=3, hidden_size=768):
        super().__init__()
        self.skip_channels = [512, 256, 64, 16]
        self.n_skip = n_skip
        head_channels = 512
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True, )
        decoder_channels = [256, 128, 64, 16]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.n_skip != 0:
            skip_channels = self.skip_channels
            for i in range(4 - self.n_skip
                           ):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels,
                                            skip_channels)
        ]
        self.blocks = nn.LayerList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.shape  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.transpose((0, 2, 1))
        x = x.reshape((B, hidden, h, w))
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


# based on https://arxiv.org/abs/2103.10504
@manager.MODELS.add_component
class TransUnet(nn.Layer):
    def __init__(self, num_classes=4):
        super().__init__()
        self.hybrid = ResNet50()
        self.transfomer = ViT_base_patch16_224()
        self.decoder = DecoderCup()
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=num_classes,
            kernel_size=3, )
        self.img_shape = [1, 224, 224]

    def forward(self, x):

        x = paddle.tile(x[:, :, 0, :], (1, 3, 1, 1))
        x, features = self.hybrid(x)
        x = self.transfomer(x)  # (B,196*768)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        logits = paddle.unsqueeze(logits, axis=2)
        return [logits, ]


if __name__ == "__main__":
    model = TransUnet()
    test_x = paddle.rand([1, 1, 224, 224])
    output = model(test_x)
    print(output.shape)

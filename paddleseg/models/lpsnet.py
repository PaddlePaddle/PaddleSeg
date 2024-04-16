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

from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils

__all__ = [
    "LPSNet",
]

_interpolate = partial(F.interpolate, mode="bilinear", align_corners=True)


@manager.MODELS.add_component
class LPSNet(nn.Layer):
    """
    The LPSNet implementation based on PaddlePaddle.

    The original article refers to
    Zhang, Yiheng and Yao, Ting and Qiu, Zhaofan and Mei, Tao. "Lightweight and Progressively-Scalable Networks for Semantic Segmentation"
    (https://arxiv.org/pdf/2207.13600)

    Args:
        depths (list): Depths of each block.
        channels (list): Channels of each block.
        scale_ratios (list): Scale ratio for each branch. The number of branches depends on length of scale_ratios.
        num_classes (int): The unique number of target classes.
        in_channels (int):  The channels of input image. Default: 3.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(
        self,
        depths,
        channels,
        scale_ratios,
        num_classes,
        in_channels=3,
        pretrained=None,
    ):
        super().__init__()

        self.depths = depths
        self.channels = channels
        self.scale_ratios = list(filter(lambda x: x > 0, scale_ratios))
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.num_paths = len(self.scale_ratios)
        self.num_blocks = len(depths)

        if self.num_blocks != len(self.channels):
            raise ValueError(
                f"Expect depths and channels have same length, but got {self.num_blocks} and {len(self.channels)}"
            )

        self.nets = nn.LayerList(
            [self._build_path() for _ in range(self.num_paths)])

        self.head = nn.Conv2D(channels[-1] * self.num_paths,
                              num_classes,
                              1,
                              bias_attr=True)

        self._init_weight(pretrained)

    def _init_weight(self, pretrained):
        if pretrained is not None:
            utils.load_entire_model(self, pretrained)

    def _build_path(self):
        path = []
        c_in = self.in_channels
        for b, (d, c) in enumerate(zip(self.depths, self.channels)):
            blocks = []
            for i in range(d):
                blocks.append(
                    layers.ConvBNReLU(
                        in_channels=c_in if i == 0 else c,
                        out_channels=c,
                        kernel_size=3,
                        padding=1,
                        stride=2 if
                        (i == 0 and b != self.num_blocks - 1) else 1,
                        bias_attr=False,
                    ))
                c_in = c
            path.append(nn.Sequential(*blocks))
        return nn.LayerList(path)

    def _preprocess_input(self, x):
        h, w = x.shape[-2:]
        return [
            _interpolate(x, (int(r * h), int(r * w))) for r in self.scale_ratios
        ]

    def forward(self, x, interact_begin_idx=2):
        input_size = x.shape[-2:]
        inputs = self._preprocess_input(x)
        feats = []
        for path, x in zip(self.nets, inputs):
            inp = x
            for idx in range(interact_begin_idx + 1):
                inp = path[idx](inp)
            feats.append(inp)

        for idx in range(interact_begin_idx + 1, self.num_blocks):
            feats = _multipath_interaction(feats)
            feats = [path[idx](x) for path, x in zip(self.nets, feats)]

        size = feats[0].shape[-2:]
        feats = [_interpolate(x, size=size) for x in feats]

        out = self.head(paddle.concat(feats, 1))

        return [_interpolate(out, size=input_size)]


def _multipath_interaction(feats):
    length = len(feats)
    if length == 1:
        return feats[0]
    sizes = [x.shape[-2:] for x in feats]
    outs = []
    looper = list(range(length))
    for i, s in enumerate(sizes):
        out = feats[i]
        for j in filter(lambda x: x != i, looper):
            out += _interpolate(feats[j], size=s)
        outs.append(out)
    return outs

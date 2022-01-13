# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.nn as nn

from paddleseg.cvlibs import manager
from paddleseg import utils


@manager.BACKBONES.add_component
class MobileNetV2(nn.Layer):
    """
        The MobileNetV2 implementation based on PaddlePaddle.

        The original article refers to
        Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
        "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
        (https://arxiv.org/abs/1801.04381).

        Args:
            channel_ratio (float, optional): The ratio of channel. Default: 1.0
            min_channel (int, optional): The minimum of channel. Default: 16
            pretrained (str, optional): The path or url of pretrained model. Default: None
        """

    def __init__(self, channel_ratio=1.0, min_channel=16, pretrained=None):
        super(MobileNetV2, self).__init__()
        self.channel_ratio = channel_ratio
        self.min_channel = min_channel
        self.pretrained = pretrained

        self.stage0 = conv_bn(3, self.depth(32), 3, 2)

        self.stage1 = InvertedResidual(self.depth(32), self.depth(16), 1, 1)

        self.stage2 = nn.Sequential(
            InvertedResidual(self.depth(16), self.depth(24), 2, 6),
            InvertedResidual(self.depth(24), self.depth(24), 1, 6),
        )

        self.stage3 = nn.Sequential(
            InvertedResidual(self.depth(24), self.depth(32), 2, 6),
            InvertedResidual(self.depth(32), self.depth(32), 1, 6),
            InvertedResidual(self.depth(32), self.depth(32), 1, 6),
        )

        self.stage4 = nn.Sequential(
            InvertedResidual(self.depth(32), self.depth(64), 2, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
        )

        self.stage5 = nn.Sequential(
            InvertedResidual(self.depth(64), self.depth(96), 1, 6),
            InvertedResidual(self.depth(96), self.depth(96), 1, 6),
            InvertedResidual(self.depth(96), self.depth(96), 1, 6),
        )

        self.stage6 = nn.Sequential(
            InvertedResidual(self.depth(96), self.depth(160), 2, 6),
            InvertedResidual(self.depth(160), self.depth(160), 1, 6),
            InvertedResidual(self.depth(160), self.depth(160), 1, 6),
        )

        self.stage7 = InvertedResidual(self.depth(160), self.depth(320), 1, 6)

        self.init_weight()

    def depth(self, channels):
        min_channel = min(channels, self.min_channel)
        return max(min_channel, int(channels * self.channel_ratio))

    def forward(self, x):
        feat_list = []

        feature_1_2 = self.stage0(x)
        feature_1_2 = self.stage1(feature_1_2)
        feature_1_4 = self.stage2(feature_1_2)
        feature_1_8 = self.stage3(feature_1_4)
        feature_1_16 = self.stage4(feature_1_8)
        feature_1_16 = self.stage5(feature_1_16)
        feature_1_32 = self.stage6(feature_1_16)
        feature_1_32 = self.stage7(feature_1_32)
        feat_list.append(feature_1_4)
        feat_list.append(feature_1_8)
        feat_list.append(feature_1_16)
        feat_list.append(feature_1_32)
        return feat_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


def conv_bn(inp, oup, kernel, stride):
    return nn.Sequential(
        nn.Conv2D(
            in_channels=inp,
            out_channels=oup,
            kernel_size=kernel,
            stride=stride,
            padding=(kernel - 1) // 2,
            bias_attr=False),
        nn.BatchNorm2D(num_features=oup, epsilon=1e-05, momentum=0.1),
        nn.ReLU())


class InvertedResidual(nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            nn.Conv2D(
                inp,
                inp * expand_ratio,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias_attr=False),
            nn.BatchNorm2D(
                num_features=inp * expand_ratio, epsilon=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(
                inp * expand_ratio,
                inp * expand_ratio,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=inp * expand_ratio,
                bias_attr=False),
            nn.BatchNorm2D(
                num_features=inp * expand_ratio, epsilon=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(
                inp * expand_ratio,
                oup,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias_attr=False),
            nn.BatchNorm2D(num_features=oup, epsilon=1e-05, momentum=0.1),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

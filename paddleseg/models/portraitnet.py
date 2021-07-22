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

from paddleseg import utils
from paddleseg.cvlibs import manager


@manager.MODELS.add_component
class PortraitNet(nn.Layer):
    """
    The PortraitNet implementation based on PaddlePaddle.

    The original article refers to
    Song-Hai Zhanga, Xin Donga, Jia Lib, Ruilong Lia, Yong-Liang Yangc
    "PortraitNet: Real-time Portrait Segmentation Network for Mobile Device"
    (https://www.yongliangyang.net/docs/mobilePotrait_c&g19.pdf).

    Args:
        num_classes (int, optional): The unique number of target classes.  Default: 2.
        backbone (Paddle.nn.Layer): Backbone network, currently support MobileNetV2.
        add_edge (bool, optional): Whether output to edge. Default: False
        pretrained (str, optional): The path or url of pretrained model. Default: None
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 min_channel=16,
                 channel_ratio=1.0,
                 add_edge=False,
                 pretrained=None):
        super(PortraitNet, self).__init__()
        self.backbone = backbone
        self.head = PortraitNetHead(num_classes, min_channel, channel_ratio,
                                    add_edge)
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        img = x[:, :3, :, :]
        img_ori = x[:, 3:, :, :]

        feat_list = self.backbone(img)
        logits_list = self.head(feat_list)

        feat_list = self.backbone(img_ori)
        logits_ori_list = self.head(feat_list)

        return [
            logits_list[0], logits_ori_list[0], logits_list[1],
            logits_ori_list[1]
        ]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class PortraitNetHead(nn.Layer):
    def __init__(self,
                 num_classes,
                 min_channel=16,
                 channel_ratio=1.0,
                 add_edge=False):
        super().__init__()
        self.min_channel = min_channel
        self.channel_ratio = channel_ratio
        self.add_edge = add_edge
        self.deconv1 = nn.Conv2DTranspose(
            self.depth(96),
            self.depth(96),
            groups=1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias_attr=False)
        self.deconv2 = nn.Conv2DTranspose(
            self.depth(32),
            self.depth(32),
            groups=1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias_attr=False)
        self.deconv3 = nn.Conv2DTranspose(
            self.depth(24),
            self.depth(24),
            groups=1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias_attr=False)
        self.deconv4 = nn.Conv2DTranspose(
            self.depth(16),
            self.depth(16),
            groups=1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias_attr=False)
        self.deconv5 = nn.Conv2DTranspose(
            self.depth(8),
            self.depth(8),
            groups=1,
            kernel_size=4,
            stride=2,
            padding=1,
            bias_attr=False)

        self.transit1 = ResidualBlock(self.depth(320), self.depth(96))
        self.transit2 = ResidualBlock(self.depth(96), self.depth(32))
        self.transit3 = ResidualBlock(self.depth(32), self.depth(24))
        self.transit4 = ResidualBlock(self.depth(24), self.depth(16))
        self.transit5 = ResidualBlock(self.depth(16), self.depth(8))

        self.pred = nn.Conv2D(
            self.depth(8), num_classes, 3, 1, 1, bias_attr=False)
        if self.add_edge:
            self.edge = nn.Conv2D(
                self.depth(8), num_classes, 3, 1, 1, bias_attr=False)

    def depth(self, channels):
        min_channel = min(channels, self.min_channel)
        return max(min_channel, int(channels * self.channel_ratio))

    def forward(self, feat_list):
        feature_1_4, feature_1_8, feature_1_16, feature_1_32 = feat_list
        up_1_16 = self.deconv1(self.transit1(feature_1_32))
        up_1_8 = self.deconv2(self.transit2(feature_1_16 + up_1_16))
        up_1_4 = self.deconv3(self.transit3(feature_1_8 + up_1_8))
        up_1_2 = self.deconv4(self.transit4(feature_1_4 + up_1_4))
        up_1_1 = self.deconv5(self.transit5(up_1_2))

        pred = self.pred(up_1_1)
        if self.add_edge:
            edge = self.edge(up_1_1)
            return pred, edge
        else:
            return pred


class ConvDw(nn.Layer):
    def __init__(self, inp, oup, kernel, stride):
        super(ConvDw, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(
                inp,
                inp,
                kernel,
                stride, (kernel - 1) // 2,
                groups=inp,
                bias_attr=False),
            nn.BatchNorm2D(num_features=inp, epsilon=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(inp, oup, 1, 1, 0, bias_attr=False),
            nn.BatchNorm2D(num_features=oup, epsilon=1e-05, momentum=0.1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Layer):
    def __init__(self, inp, oup, stride=1):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            ConvDw(inp, oup, 3, stride=stride),
            nn.Conv2D(
                in_channels=oup,
                out_channels=oup,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=oup,
                bias_attr=False),
            nn.BatchNorm2D(num_features=oup, epsilon=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(
                in_channels=oup,
                out_channels=oup,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False),
            nn.BatchNorm2D(num_features=oup, epsilon=1e-05, momentum=0.1),
        )
        if inp == oup:
            self.residual = None
        else:
            self.residual = nn.Sequential(
                nn.Conv2D(
                    in_channels=inp,
                    out_channels=oup,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias_attr=False),
                nn.BatchNorm2D(num_features=oup, epsilon=1e-05, momentum=0.1),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.block(x)
        if self.residual is not None:
            residual = self.residual(x)

        out += residual
        out = self.relu(out)
        return out

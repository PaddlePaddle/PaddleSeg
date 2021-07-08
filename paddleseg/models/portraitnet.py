# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import numpy as np
from paddleseg.cvlibs import manager, param_init
from paddleseg import utils


def make_bilinear_weights(size, num_channels, in_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    w = np.zeros([num_channels, in_channels, size, size])
    for i in range(num_channels):
        for j in range(in_channels):
            w[i, j] = filt
    return w


def conv_1x1(inp, oup):
    """1x1 convolution with padding"""
    return nn.Conv2D(in_channels=inp,
                     out_channels=oup,
                     kernel_size=1,
                     stride=1,
                     padding=0,
                     bias_attr=False)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2D(in_channels=inp,
                  out_channels=oup,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  bias_attr=False),
        nn.BatchNorm2D(num_features=oup, epsilon=1e-05, momentum=0.1),
        nn.ReLU())


def conv_bn(inp, oup, kernel, stride):
    return nn.Sequential(
        nn.Conv2D(in_channels=inp,
                  out_channels=oup,
                  kernel_size=kernel,
                  stride=stride,
                  padding=(kernel - 1) // 2,
                  bias_attr=False),
        nn.BatchNorm2D(num_features=oup, epsilon=1e-05, momentum=0.1),
        nn.ReLU())


def conv_dw(inp, oup, kernel, stride):
    return nn.Sequential(
        nn.Conv2D(inp,
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


class InvertedResidual(nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            nn.Conv2D(inp,
                      inp * expand_ratio,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      dilation=1,
                      groups=1,
                      bias_attr=False),
            nn.BatchNorm2D(num_features=inp * expand_ratio,
                           epsilon=1e-05,
                           momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(inp * expand_ratio,
                      inp * expand_ratio,
                      kernel_size=3,
                      stride=stride,
                      padding=dilation,
                      dilation=dilation,
                      groups=inp * expand_ratio,
                      bias_attr=False),
            nn.BatchNorm2D(num_features=inp * expand_ratio,
                           epsilon=1e-05,
                           momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(inp * expand_ratio,
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


class ResidualBlock(nn.Layer):
    def __init__(self, inp, oup, stride=1):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            conv_dw(inp, oup, 3, stride=stride),
            nn.Conv2D(in_channels=oup,
                      out_channels=oup,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=oup,
                      bias_attr=False),
            nn.BatchNorm2D(num_features=oup, epsilon=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Conv2D(in_channels=oup,
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
                nn.Conv2D(in_channels=inp,
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


@manager.MODELS.add_component
class MobileNetV2(nn.Layer):
    def __init__(self,
                 num_classes=2,
                 add_edge=False,
                 channel_ratio=1.0,
                 min_channel=16,
                 weight_init=True,
                 pretrained=None):
        super(MobileNetV2, self).__init__()
        self.add_edge = add_edge
        self.channel_ratio = channel_ratio
        self.min_channel = min_channel
        self.pretrained = pretrained

        self.stage0 = conv_bn(3, self.depth(32), 3, 2)

        self.stage1 = InvertedResidual(self.depth(32), self.depth(16), 1,
                                       1)  # 1/2

        self.stage2 = nn.Sequential(  # 1/4
            InvertedResidual(self.depth(16), self.depth(24), 2, 6),
            InvertedResidual(self.depth(24), self.depth(24), 1, 6),
        )

        self.stage3 = nn.Sequential(  # 1/8
            InvertedResidual(self.depth(24), self.depth(32), 2, 6),
            InvertedResidual(self.depth(32), self.depth(32), 1, 6),
            InvertedResidual(self.depth(32), self.depth(32), 1, 6),
        )

        self.stage4 = nn.Sequential(  # 1/16
            InvertedResidual(self.depth(32), self.depth(64), 2, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
            InvertedResidual(self.depth(64), self.depth(64), 1, 6),
        )

        self.stage5 = nn.Sequential(  # 1/16
            InvertedResidual(self.depth(64), self.depth(96), 1, 6),
            InvertedResidual(self.depth(96), self.depth(96), 1, 6),
            InvertedResidual(self.depth(96), self.depth(96), 1, 6),
        )

        self.stage6 = nn.Sequential(  # 1/32
            InvertedResidual(self.depth(96), self.depth(160), 2, 6),
            InvertedResidual(self.depth(160), self.depth(160), 1, 6),
            InvertedResidual(self.depth(160), self.depth(160), 1, 6),
        )

        self.stage7 = InvertedResidual(self.depth(160), self.depth(320), 1,
                                       6)  # 1/32

        self.deconv1 = nn.Conv2DTranspose(self.depth(96),
                                          self.depth(96),
                                          groups=1,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias_attr=False)
        self.deconv2 = nn.Conv2DTranspose(self.depth(32),
                                          self.depth(32),
                                          groups=1,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias_attr=False)
        self.deconv3 = nn.Conv2DTranspose(self.depth(24),
                                          self.depth(24),
                                          groups=1,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias_attr=False)
        self.deconv4 = nn.Conv2DTranspose(self.depth(16),
                                          self.depth(16),
                                          groups=1,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias_attr=False)
        self.deconv5 = nn.Conv2DTranspose(self.depth(8),
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

        self.pred = nn.Conv2D(self.depth(8),
                              num_classes,
                              3,
                              1,
                              1,
                              bias_attr=False)
        if self.add_edge:
            self.edge = nn.Conv2D(self.depth(8),
                                  num_classes,
                                  3,
                                  1,
                                  1,
                                  bias_attr=False)

        if weight_init:
            self._initialize_weights()

    def depth(self, channels):
        min_channel = min(channels, self.min_channel)
        return max(min_channel, int(channels * self.channel_ratio))

    def forward(self, x):
        feature_1_2 = self.stage0(x)
        feature_1_2 = self.stage1(feature_1_2)
        feature_1_4 = self.stage2(feature_1_2)
        feature_1_8 = self.stage3(feature_1_4)
        feature_1_16 = self.stage4(feature_1_8)
        feature_1_16 = self.stage5(feature_1_16)
        feature_1_32 = self.stage6(feature_1_16)
        feature_1_32 = self.stage7(feature_1_32)

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

    def _initialize_weights(self):
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                param_init.kaiming_normal_init(sublayer.weight)
                if sublayer.bias is not None:
                    param_init.constant_init(sublayer.bias, value=0.0)
            elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(sublayer.weight, value=1.0)
                param_init.constant_init(sublayer.bias, value=0.0)
            elif isinstance(sublayer, nn.Linear):
                param_init.normal_init(sublayer.weight, loc=0, scale=0.01)
                param_init.constant_init(sublayer.bias, value=0.0)
            elif isinstance(sublayer, nn.Conv2DTranspose):
                initial_weight = make_bilinear_weights(
                    sublayer._kernel_size[0], sublayer.weight.shape[0],
                    sublayer.weight.shape[1])  # same as caffe
                initializer = nn.initializer.Assign(initial_weight)
                initializer(sublayer.weight, sublayer.weight.block)
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


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
        add_edge (bool, optional): Whether output to edge. Default: False
        channel_ratio (float, optional): The ratio of channel. Default: 1.0
        min_channel (int, optional): The minimum of channel. Default: 16
        weight_init (bool, optional): Whether to initialize parameters. Default: True
        pretrained (str, optional): The path or url of pretrained model. Default: None
    """
    def __init__(self,
                 num_classes=2,
                 add_edge=False,
                 channel_ratio=1.0,
                 min_channel=16,
                 weight_init=True,
                 pretrained=None):
        super(PortraitNet, self).__init__()
        self.model = MobileNetV2(num_classes=num_classes,
                                 add_edge=add_edge,
                                 channel_ratio=channel_ratio,
                                 min_channel=min_channel,
                                 weight_init=weight_init,
                                 pretrained=pretrained)

    def forward(self, x):
        img = x[:, :3, :, :]
        img_ori = x[:, 3:, :, :]
        logits_list = self.model(img)
        logits_ori_list = self.model(img_ori)
        return [
            logits_list[0], logits_ori_list[0], logits_list[1],
            logits_ori_list[1]
        ]

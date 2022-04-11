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
import paddle.nn.functional as F

from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.utils import utils


class DualResNet(nn.Layer):
    """
    The DDRNet implementation based on PaddlePaddle.

    The original article refers to
    Yuanduo Hong, Huihui Pan, Weichao Sun, et al. "Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes"
    (https://arxiv.org/abs/2101.06085)

    Args:
        num_classes (int): The unique number of target classes.
        in_channels (int, optional): Number of input channels. Default: 3.
        block_layers (list, tuple): The numbers of layers in different blocks. Default: [2, 2, 2, 2].
        planes (int): Base channels in network. Default: 64.
        spp_planes (int): Branch channels for DAPPM. Default: 128.
        head_planes (int): Mid channels of segmentation head. Default: 128.
        enable_auxiliary_loss (bool): Whether use auxiliary head for stage3. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 in_channels=3,
                 block_layers=[2, 2, 2, 2],
                 planes=64,
                 spp_planes=128,
                 head_planes=128,
                 enable_auxiliary_loss=False,
                 pretrained=None):
        super().__init__()
        highres_planes = planes * 2
        self.enable_auxiliary_loss = enable_auxiliary_loss
        self.conv1 = nn.Sequential(
            layers.ConvBNReLU(
                in_channels, planes, kernel_size=3, stride=2, padding=1),
            layers.ConvBNReLU(
                planes, planes, kernel_size=3, stride=2, padding=1), )
        self.relu = nn.ReLU()
        self.layer1 = self._make_layers(BasicBlock, planes, planes,
                                        block_layers[0])
        self.layer2 = self._make_layers(
            BasicBlock, planes, planes * 2, block_layers[1], stride=2)
        self.layer3 = self._make_layers(
            BasicBlock, planes * 2, planes * 4, block_layers[2], stride=2)
        self.layer4 = self._make_layers(
            BasicBlock, planes * 4, planes * 8, block_layers[3], stride=2)

        self.compression3 = layers.ConvBN(
            planes * 4, highres_planes, kernel_size=1, bias_attr=False)

        self.compression4 = layers.ConvBN(
            planes * 8, highres_planes, kernel_size=1, bias_attr=False)

        self.down3 = layers.ConvBN(
            highres_planes,
            planes * 4,
            kernel_size=3,
            stride=2,
            bias_attr=False)

        self.down4 = nn.Sequential(
            layers.ConvBNReLU(
                highres_planes,
                planes * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias_attr=False),
            layers.ConvBN(
                planes * 4,
                planes * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias_attr=False))

        self.layer3_ = self._make_layers(BasicBlock, planes * 2, highres_planes,
                                         2)
        self.layer4_ = self._make_layers(BasicBlock, highres_planes,
                                         highres_planes, 2)
        self.layer5_ = self._make_layers(Bottleneck, highres_planes,
                                         highres_planes, 1)
        self.layer5 = self._make_layers(
            Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)
        if self.enable_auxiliary_loss:
            self.aux_head = DDRNetHead(highres_planes, head_planes, num_classes)
        self.head = DDRNetHead(planes * 4, head_planes, num_classes)

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
        else:
            for m in self.sublayers():
                if isinstance(m, nn.Conv2D):
                    param_init.kaiming_normal_init(m.weight)
                elif isinstance(m, nn.BatchNorm2D):
                    param_init.constant_init(m.weight, value=1)
                    param_init.constant_init(m.bias, value=0)

    def _make_layers(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion), )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        n, c, h, w = paddle.shape(x)
        width_output = w // 8
        height_output = h // 8

        x = self.conv1(x)
        stage1_out = self.layer1(x)
        stage2_out = self.layer2(self.relu(stage1_out))
        stage3_out = self.layer3(self.relu(stage2_out))
        stage3_out_dual = self.layer3_(self.relu(stage2_out))
        x = stage3_out + self.down3(self.relu(stage3_out_dual))
        stage3_merge = stage3_out_dual + F.interpolate(
            self.compression3(self.relu(stage3_out)),
            size=[height_output, width_output],
            mode='bilinear')

        stage4_out = self.layer4(self.relu(x))
        stage4_out_dual = self.layer4_(self.relu(stage3_merge))

        x = stage4_out + self.down4(self.relu(stage4_out_dual))
        stage4_merge = stage4_out_dual + F.interpolate(
            self.compression4(self.relu(stage4_out)),
            size=[height_output, width_output],
            mode='bilinear')

        stage5_out_dual = self.layer5_(self.relu(stage4_merge))
        x = F.interpolate(
            self.spp(self.layer5(self.relu(x))),
            size=[height_output, width_output],
            mode='bilinear')

        output = self.head(x + stage5_out_dual)
        logit_list = []
        logit_list.append(output)

        if self.enable_auxiliary_loss:
            aux_out = self.aux_head(stage3_merge)
            logit_list.append(aux_out)
        return [
            F.interpolate(
                logit, [h, w], mode='bilinear') for logit in logit_list
        ]


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 no_relu=False):
        super().__init__()
        self.conv_bn_relu = layers.ConvBNReLU(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias_attr=False)
        self.relu = nn.ReLU()
        self.conv_bn = layers.ConvBN(
            planes, planes, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv_bn_relu(x)
        out = self.conv_bn(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Layer):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 no_relu=True):
        super().__init__()
        self.conv_bn_relu1 = layers.ConvBNReLU(
            inplanes, planes, kernel_size=1, bias_attr=False)
        self.conv_bn_relu2 = layers.ConvBNReLU(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias_attr=False)
        self.conv_bn = layers.ConvBN(
            planes, planes * self.expansion, kernel_size=1, bias_attr=False)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv_bn_relu1(x)
        out = self.conv_bn_relu2(out)
        out = self.conv_bn(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class DAPPM(nn.Layer):
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=5, stride=2, padding=2),
            layers.SyncBatchNorm(inplanes),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        self.scale2 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=9, stride=4, padding=4),
            layers.SyncBatchNorm(inplanes),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        self.scale3 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=17, stride=8, padding=8),
            layers.SyncBatchNorm(inplanes),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            layers.SyncBatchNorm(inplanes),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        self.scale0 = nn.Sequential(
            layers.SyncBatchNorm(inplanes),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )
        self.process1 = nn.Sequential(
            layers.SyncBatchNorm(branch_planes),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes,
                branch_planes,
                kernel_size=3,
                padding=1,
                bias_attr=False), )
        self.process2 = nn.Sequential(
            layers.SyncBatchNorm(branch_planes),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes,
                branch_planes,
                kernel_size=3,
                padding=1,
                bias_attr=False), )
        self.process3 = nn.Sequential(
            layers.SyncBatchNorm(branch_planes),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes,
                branch_planes,
                kernel_size=3,
                padding=1,
                bias_attr=False), )
        self.process4 = nn.Sequential(
            layers.SyncBatchNorm(branch_planes),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes,
                branch_planes,
                kernel_size=3,
                padding=1,
                bias_attr=False), )
        self.compression = nn.Sequential(
            layers.SyncBatchNorm(branch_planes * 5),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes * 5, outplanes, kernel_size=1, bias_attr=False))
        self.shortcut = nn.Sequential(
            layers.SyncBatchNorm(inplanes),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, outplanes, kernel_size=1, bias_attr=False))

    def forward(self, x):
        n, c, h, w = paddle.shape(x)
        x0 = self.scale0(x)
        x1 = self.process1(
            F.interpolate(
                self.scale1(x), size=[h, w], mode='bilinear') + x0)
        x2 = self.process2(
            F.interpolate(
                self.scale2(x), size=[h, w], mode='bilinear') + x1)
        x3 = self.process3(
            F.interpolate(
                self.scale3(x), size=[h, w], mode='bilinear') + x2)
        x4 = self.process4(
            F.interpolate(
                self.scale4(x), size=[h, w], mode='bilinear') + x3)

        out = self.compression(paddle.concat([x0, x1, x2, x3, x4],
                                             1)) + self.shortcut(x)
        return out


class DDRNetHead(nn.Layer):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2D(inplanes)
        self.relu = nn.ReLU()
        self.conv_bn_relu = layers.ConvBNReLU(
            inplanes, interplanes, kernel_size=3, padding=1, bias_attr=False)
        self.conv = nn.Conv2D(
            interplanes, outplanes, kernel_size=1, padding=0, bias_attr=True)

        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_bn_relu(x)
        out = self.conv(x)

        if self.scale_factor is not None:
            out = F.interpolate(
                out, scale_factor=self.scale_factor, mode='bilinear')
        return out


@manager.MODELS.add_component
def DDRNet_23(**kwargs):
    return DualResNet(
        block_layers=[2, 2, 2, 2],
        planes=64,
        spp_planes=128,
        head_planes=128,
        **kwargs)

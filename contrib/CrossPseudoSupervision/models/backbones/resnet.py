# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddleseg.utils import utils
from paddleseg.models import layers
from paddleseg.cvlibs import manager

__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 norm_layer=None,
                 bn_eps=1e-5,
                 bn_momentum=0.1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, epsilon=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU()
        self.relu_inplace = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, epsilon=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 norm_layer=None,
                 bn_eps=1e-5,
                 bn_momentum=0.1,
                 downsample=None,
                 dilate=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = norm_layer(planes, epsilon=bn_eps, momentum=bn_momentum)

        if dilate is not None:
            self.conv2 = nn.Conv2D(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=dilate,
                dilation=dilate,
                bias_attr=False)
        else:
            self.conv2 = nn.Conv2D(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias_attr=False)

        self.bn2 = norm_layer(planes, epsilon=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2D(
            planes, planes * self.expansion, kernel_size=1, bias_attr=False)
        self.bn3 = norm_layer(
            planes * self.expansion, epsilon=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU()
        self.relu_inplace = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class ResNet(nn.Layer):
    def __init__(self,
                 layer,
                 block,
                 pretrained=None,
                 norm_layer=layers.SyncBatchNorm,
                 bn_eps=1e-5,
                 bn_momentum=0.1,
                 deep_stem=False,
                 stem_width=32,
                 as_backbone=True,
                 in_channels=3):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNet, self).__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2D(
                    in_channels,
                    stem_width,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias_attr=False),
                norm_layer(
                    stem_width, epsilon=bn_eps, momentum=bn_momentum),
                nn.ReLU(),
                nn.Conv2D(
                    stem_width,
                    stem_width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias_attr=False),
                norm_layer(
                    stem_width, epsilon=bn_eps, momentum=bn_momentum),
                nn.ReLU(),
                nn.Conv2D(
                    stem_width,
                    stem_width * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias_attr=False), )
        else:
            self.conv1 = nn.Conv2D(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias_attr=False)

        num_filters = [64, 128, 256, 512]

        if layer == 18:
            depth = [2, 2, 2, 2]
        elif layer == 34 or layer == 50:
            depth = [3, 4, 6, 3]
        elif layer == 101:
            depth = [3, 4, 23, 3]
        elif layer == 152:
            depth = [3, 8, 36, 3]
        elif layer == 200:
            depth = [3, 12, 48, 3]

        # for channels of four returned stages
        self.feat_channels = [c * 4 for c in num_filters
                              ] if layer >= 50 else num_filters

        self.bn1 = norm_layer(
            stem_width * 2 if deep_stem else 64,
            epsilon=bn_eps,
            momentum=bn_momentum)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block,
            norm_layer,
            64,
            depth[0],
            bn_eps=bn_eps,
            bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(
            block,
            norm_layer,
            128,
            depth[1],
            stride=2,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(
            block,
            norm_layer,
            256,
            depth[2],
            stride=2,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum)

        if as_backbone:
            self.layer4 = self._make_layer__nostride_dilate(
                block,
                norm_layer,
                512,
                depth[3],
                stride=1,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum)
        else:
            self.layer4 = self._make_layer(
                block,
                norm_layer,
                512,
                depth[3],
                stride=2,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum)

        self.pretrained = pretrained
        self.init_weight()

    def _make_layer(self,
                    block,
                    norm_layer,
                    planes,
                    blocks,
                    stride=1,
                    bn_eps=1e-5,
                    bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False),
                norm_layer(
                    planes * block.expansion,
                    epsilon=bn_eps,
                    momentum=bn_momentum), )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, norm_layer, bn_eps,
                  bn_momentum, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                    bn_eps=bn_eps,
                    bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def _make_layer__nostride_dilate(self,
                                     block,
                                     norm_layer,
                                     planes,
                                     blocks,
                                     stride=1,
                                     bn_eps=1e-5,
                                     bn_momentum=0.1):
        downsample = nn.Sequential(
            nn.Conv2D(
                self.inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=1,
                bias_attr=False),
            norm_layer(
                planes * block.expansion, epsilon=bn_eps, momentum=bn_momentum),
        )

        dilate = 2
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, norm_layer, bn_eps,
                  bn_momentum, downsample, dilate))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            dilate *= 2
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                    bn_eps=bn_eps,
                    bn_momentum=bn_momentum,
                    dilate=dilate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        blocks = []
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        x = self.layer4(x)
        blocks.append(x)

        return blocks

    def init_weight(self):
        utils.load_pretrained_model(self, self.pretrained)


@manager.BACKBONES.add_component
def ResNet18(**args):
    model = ResNet(layer=18, block=BasicBlock, **args)
    return model


def ResNet34(**args):
    model = ResNet(layer=34, block=BasicBlock, **args)
    return model


@manager.BACKBONES.add_component
def ResNet50(**args):
    model = ResNet(layer=50, block=Bottleneck, **args)
    return model


@manager.BACKBONES.add_component
def ResNet101(**args):
    model = ResNet(layer=101, block=Bottleneck, **args)
    return model


def ResNet152(**args):
    model = ResNet(layer=152, block=Bottleneck, **args)
    return model

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

# The gca code was heavily based on https://github.com/Yaoyi-Li/GCA-Matting
# and https://github.com/open-mmlab/mmediting

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import manager, param_init
from paddleseg.utils import utils

from ppmatting.models.layers import GuidedCxtAtten


class ResNet_D(nn.Layer):
    def __init__(self,
                 input_channels,
                 layers,
                 late_downsample=False,
                 pretrained=None):

        super().__init__()

        self.pretrained = pretrained

        self._norm_layer = nn.BatchNorm
        self.inplanes = 64
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32
        self.start_stride = [1, 2, 1, 2] if late_downsample else [2, 1, 2, 1]
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2D(
                input_channels,
                32,
                kernel_size=3,
                stride=self.start_stride[0],
                padding=1,
                bias_attr=False))
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2D(
                32,
                self.midplanes,
                kernel_size=3,
                stride=self.start_stride[1],
                padding=1,
                bias_attr=False))
        self.conv3 = nn.utils.spectral_norm(
            nn.Conv2D(
                self.midplanes,
                self.inplanes,
                kernel_size=3,
                stride=self.start_stride[2],
                padding=1,
                bias_attr=False))
        self.bn1 = self._norm_layer(32)
        self.bn2 = self._norm_layer(self.midplanes)
        self.bn3 = self._norm_layer(self.inplanes)
        self.activation = nn.ReLU()
        self.layer1 = self._make_layer(
            BasicBlock, 64, layers[0], stride=self.start_stride[3])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer_bottleneck = self._make_layer(
            BasicBlock, 512, layers[3], stride=2)

        self.init_weight()

    def _make_layer(self, block, planes, block_num, stride=1):
        if block_num == 0:
            return nn.Sequential(nn.Identity())
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2D(2, stride),
                nn.utils.spectral_norm(
                    conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion), )
        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.utils.spectral_norm(
                    conv1x1(self.inplanes, planes * block.expansion, stride)),
                norm_layer(planes * block.expansion), )

        layers = [block(self.inplanes, planes, stride, downsample, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x1 = self.activation(x)  # N x 32 x 256 x 256
        x = self.conv3(x1)
        x = self.bn3(x)
        x2 = self.activation(x)  # N x 64 x 128 x 128

        x3 = self.layer1(x2)  # N x 64 x 128 x 128
        x4 = self.layer2(x3)  # N x 128 x 64 x 64
        x5 = self.layer3(x4)  # N x 256 x 32 x 32
        x = self.layer_bottleneck(x5)  # N x 512 x 16 x 16

        return x, (x1, x2, x3, x4, x5)

    def init_weight(self):

        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):

                if hasattr(layer, "weight_orig"):
                    param = layer.weight_orig
                else:
                    param = layer.weight
                param_init.xavier_uniform(param)

            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)

            elif isinstance(layer, BasicBlock):
                param_init.constant_init(layer.bn2.weight, value=0.0)

        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)


@manager.MODELS.add_component
class ResShortCut_D(ResNet_D):
    def __init__(self,
                 input_channels,
                 layers,
                 late_downsample=False,
                 pretrained=None):
        super().__init__(
            input_channels,
            layers,
            late_downsample=late_downsample,
            pretrained=pretrained)

        self.shortcut_inplane = [input_channels, self.midplanes, 64, 128, 256]
        self.shortcut_plane = [32, self.midplanes, 64, 128, 256]

        self.shortcut = nn.LayerList()
        for stage, inplane in enumerate(self.shortcut_inplane):
            self.shortcut.append(
                self._make_shortcut(inplane, self.shortcut_plane[stage]))

    def _make_shortcut(self, inplane, planes):
        return nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2D(
                    inplane, planes, kernel_size=3, padding=1,
                    bias_attr=False)),
            nn.ReLU(),
            self._norm_layer(planes),
            nn.utils.spectral_norm(
                nn.Conv2D(
                    planes, planes, kernel_size=3, padding=1, bias_attr=False)),
            nn.ReLU(),
            self._norm_layer(planes))

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        x1 = self.activation(out)  # N x 32 x 256 x 256
        out = self.conv3(x1)
        out = self.bn3(out)
        out = self.activation(out)

        x2 = self.layer1(out)  # N x 64 x 128 x 128
        x3 = self.layer2(x2)  # N x 128 x 64 x 64
        x4 = self.layer3(x3)  # N x 256 x 32 x 32
        out = self.layer_bottleneck(x4)  # N x 512 x 16 x 16

        fea1 = self.shortcut[0](x)  # input image and trimap
        fea2 = self.shortcut[1](x1)
        fea3 = self.shortcut[2](x2)
        fea4 = self.shortcut[3](x3)
        fea5 = self.shortcut[4](x4)

        return out, {
            'shortcut': (fea1, fea2, fea3, fea4, fea5),
            'image': x[:, :3, ...]
        }


@manager.MODELS.add_component
class ResGuidedCxtAtten(ResNet_D):
    def __init__(self,
                 input_channels,
                 layers,
                 late_downsample=False,
                 pretrained=None):
        super().__init__(
            input_channels,
            layers,
            late_downsample=late_downsample,
            pretrained=pretrained)
        self.input_channels = input_channels
        self.shortcut_inplane = [input_channels, self.midplanes, 64, 128, 256]
        self.shortcut_plane = [32, self.midplanes, 64, 128, 256]

        self.shortcut = nn.LayerList()
        for stage, inplane in enumerate(self.shortcut_inplane):
            self.shortcut.append(
                self._make_shortcut(inplane, self.shortcut_plane[stage]))

        self.guidance_head = nn.Sequential(
            nn.Pad2D(
                1, mode="reflect"),
            nn.utils.spectral_norm(
                nn.Conv2D(
                    3, 16, kernel_size=3, padding=0, stride=2,
                    bias_attr=False)),
            nn.ReLU(),
            self._norm_layer(16),
            nn.Pad2D(
                1, mode="reflect"),
            nn.utils.spectral_norm(
                nn.Conv2D(
                    16, 32, kernel_size=3, padding=0, stride=2,
                    bias_attr=False)),
            nn.ReLU(),
            self._norm_layer(32),
            nn.Pad2D(
                1, mode="reflect"),
            nn.utils.spectral_norm(
                nn.Conv2D(
                    32,
                    128,
                    kernel_size=3,
                    padding=0,
                    stride=2,
                    bias_attr=False)),
            nn.ReLU(),
            self._norm_layer(128))

        self.gca = GuidedCxtAtten(128, 128)

        self.init_weight()

    def init_weight(self):

        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                initializer = nn.initializer.XavierUniform()
                if hasattr(layer, "weight_orig"):
                    param = layer.weight_orig
                else:
                    param = layer.weight
                initializer(param, param.block)

            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)

            elif isinstance(layer, BasicBlock):
                param_init.constant_init(layer.bn2.weight, value=0.0)

        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)

    def _make_shortcut(self, inplane, planes):
        return nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2D(
                    inplane, planes, kernel_size=3, padding=1,
                    bias_attr=False)),
            nn.ReLU(),
            self._norm_layer(planes),
            nn.utils.spectral_norm(
                nn.Conv2D(
                    planes, planes, kernel_size=3, padding=1, bias_attr=False)),
            nn.ReLU(),
            self._norm_layer(planes))

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        x1 = self.activation(out)  # N x 32 x 256 x 256
        out = self.conv3(x1)
        out = self.bn3(out)
        out = self.activation(out)

        im_fea = self.guidance_head(
            x[:, :3, ...])  # downsample origin image and extract features
        if self.input_channels == 6:
            unknown = F.interpolate(
                x[:, 4:5, ...], scale_factor=1 / 8, mode='nearest')
        else:
            unknown = x[:, 3:, ...].equal(paddle.to_tensor([1.]))
            unknown = paddle.cast(unknown, dtype='float32')
            unknown = F.interpolate(unknown, scale_factor=1 / 8, mode='nearest')

        x2 = self.layer1(out)  # N x 64 x 128 x 128
        x3 = self.layer2(x2)  # N x 128 x 64 x 64
        x3 = self.gca(im_fea, x3, unknown)  # contextual attention
        x4 = self.layer3(x3)  # N x 256 x 32 x 32
        out = self.layer_bottleneck(x4)  # N x 512 x 16 x 16

        fea1 = self.shortcut[0](x)  # input image and trimap
        fea2 = self.shortcut[1](x1)
        fea3 = self.shortcut[2](x2)
        fea4 = self.shortcut[3](x3)
        fea5 = self.shortcut[4](x4)

        return out, {
            'shortcut': (fea1, fea2, fea3, fea4, fea5),
            'image_fea': im_fea,
            'unknown': unknown,
        }


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.utils.spectral_norm(conv3x3(inplanes, planes, stride))
        self.bn1 = norm_layer(planes)
        self.activation = nn.ReLU()
        self.conv2 = nn.utils.spectral_norm(conv3x3(planes, planes))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias_attr=False,
        dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(
        in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)

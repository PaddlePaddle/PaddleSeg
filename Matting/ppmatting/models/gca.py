# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
from paddleseg.models import layers
from paddleseg import utils
from paddleseg.cvlibs import manager, param_init

from ppmatting.models.layers import GuidedCxtAtten


@manager.MODELS.add_component
class GCABaseline(nn.Layer):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.encoder = backbone
        self.decoder = ResShortCut_D_Dec([2, 3, 3, 2])

    def forward(self, inputs):

        x = paddle.concat([inputs['img'], inputs['trimap'] / 255], axis=1)
        embedding, mid_fea = self.encoder(x)
        alpha_pred = self.decoder(embedding, mid_fea)

        if self.training:
            logit_dict = {'alpha_pred': alpha_pred, }
            loss_dict = {}
            alpha_gt = inputs['alpha']
            loss_dict["alpha"] = F.l1_loss(alpha_pred, alpha_gt)
            loss_dict["all"] = loss_dict["alpha"]
            return logit_dict, loss_dict

        return alpha_pred


@manager.MODELS.add_component
class GCA(GCABaseline):
    def __init__(self, backbone, pretrained=None):
        super().__init__(backbone, pretrained)
        self.decoder = ResGuidedCxtAtten_Dec([2, 3, 3, 2])


def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5x5 convolution with padding"""
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=5,
        stride=stride,
        padding=2,
        groups=groups,
        bias_attr=False,
        dilation=dilation)


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


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 upsample=None,
                 norm_layer=None,
                 large_kernel=False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm
        self.stride = stride
        conv = conv5x5 if large_kernel else conv3x3
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if self.stride > 1:
            self.conv1 = nn.utils.spectral_norm(
                nn.Conv2DTranspose(
                    inplanes,
                    inplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias_attr=False))
        else:
            self.conv1 = nn.utils.spectral_norm(conv(inplanes, inplanes))
        self.bn1 = norm_layer(inplanes)
        self.activation = nn.LeakyReLU(0.2)
        self.conv2 = nn.utils.spectral_norm(conv(inplanes, planes))
        self.bn2 = norm_layer(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.activation(out)

        return out


class ResNet_D_Dec(nn.Layer):
    def __init__(self,
                 layers=[3, 4, 4, 2],
                 norm_layer=None,
                 large_kernel=False,
                 late_downsample=False):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm
        self._norm_layer = norm_layer
        self.large_kernel = large_kernel
        self.kernel_size = 5 if self.large_kernel else 3

        self.inplanes = 512 if layers[0] > 0 else 256
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32

        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2DTranspose(
                self.midplanes,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias_attr=False))
        self.bn1 = norm_layer(32)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2D(
            32,
            1,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2)
        self.upsample = nn.UpsamplingNearest2D(scale_factor=2)
        self.tanh = nn.Tanh()
        self.layer1 = self._make_layer(BasicBlock, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(
            BasicBlock, self.midplanes, layers[3], stride=2)

        self.init_weight()

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks == 0:
            return nn.Sequential(nn.Identity())
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.UpsamplingNearest2D(scale_factor=2),
                nn.utils.spectral_norm(
                    conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion), )
        elif self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.utils.spectral_norm(
                    conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion), )

        layers = [
            block(self.inplanes, planes, stride, upsample, norm_layer,
                  self.large_kernel)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                    large_kernel=self.large_kernel))

        return nn.Sequential(*layers)

    def forward(self, x, mid_fea):
        x = self.layer1(x)  # N x 256 x 32 x 32
        print(x.shape)
        x = self.layer2(x)  # N x 128 x 64 x 64
        print(x.shape)
        x = self.layer3(x)  # N x 64 x 128 x 128
        print(x.shape)
        x = self.layer4(x)  # N x 32 x 256 x 256
        print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)

        alpha = (self.tanh(x) + 1.0) / 2.0

        return alpha

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


class ResShortCut_D_Dec(ResNet_D_Dec):
    def __init__(self,
                 layers,
                 norm_layer=None,
                 large_kernel=False,
                 late_downsample=False):
        super().__init__(
            layers, norm_layer, large_kernel, late_downsample=late_downsample)

    def forward(self, x, mid_fea):
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        x = self.layer3(x) + fea3
        x = self.layer4(x) + fea2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x = self.conv2(x)

        alpha = (self.tanh(x) + 1.0) / 2.0

        return alpha


class ResGuidedCxtAtten_Dec(ResNet_D_Dec):
    def __init__(self,
                 layers,
                 norm_layer=None,
                 large_kernel=False,
                 late_downsample=False):
        super().__init__(
            layers, norm_layer, large_kernel, late_downsample=late_downsample)
        self.gca = GuidedCxtAtten(128, 128)

    def forward(self, x, mid_fea):
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        im = mid_fea['image_fea']
        x = self.layer1(x) + fea5  # N x 256 x 32 x 32
        x = self.layer2(x) + fea4  # N x 128 x 64 x 64
        x = self.gca(im, x, mid_fea['unknown'])  # contextual attention
        x = self.layer3(x) + fea3  # N x 64 x 128 x 128
        x = self.layer4(x) + fea2  # N x 32 x 256 x 256
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x = self.conv2(x)

        alpha = (self.tanh(x) + 1.0) / 2.0

        return alpha

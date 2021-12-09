# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils

__all__ = ["ResNet101"]


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.padding = dilation
        self.stride = stride
        self.downsample = downsample

        self.conv1 = nn.Conv2D(
            inplanes, planes, kernel_size=1, stride=stride, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)

        self.conv2 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=self.padding,
            bias_attr=False,
            dilation=dilation)
        self.bn2 = nn.BatchNorm2D(planes)

        self.conv3 = nn.Conv2D(
            planes, planes * 4, bias_attr=False, kernel_size=1)
        self.bn3 = nn.BatchNorm2D(planes * 4)

        self.relu = nn.ReLU()

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

        out += residual
        out = self.relu(out)

        return out


class ClassifierModule(nn.Layer):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.LayerList()
        for dilation, padding in zip(dilation_series, padding_series):
            weight_attr = paddle.ParamAttr(
                initializer=nn.initializer.Normal(std=0.01), learning_rate=10.0)
            bias_attr = paddle.ParamAttr(
                initializer=nn.initializer.Constant(value=0.0),
                learning_rate=10.0)
            self.conv2d_list.append(
                nn.Conv2D(
                    inplanes,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    weight_attr=weight_attr,
                    bias_attr=bias_attr))

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class ResNetMulti(nn.Layer):
    def __init__(self, block, num_layers, num_classes):
        super(ResNetMulti, self).__init__()
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=1),
            learning_rate=0.1)
        self.bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0))
        self.weight_attr_conv = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Normal(std=0.01))
        self.inplanes = 64
        self.conv1 = nn.Conv2D(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias_attr=False,
            weight_attr=self.weight_attr_conv)
        self.bn1 = nn.BatchNorm2D(
            64, bias_attr=self.bias_attr, weight_attr=self.weight_attr)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, num_layers[0])
        self.layer2 = self._make_layer(block, 128, num_layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, 256, num_layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, num_layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(ClassifierModule, 1024,
                                            [6, 12, 18, 24], [6, 12, 18, 24],
                                            num_classes)
        self.layer6 = self._make_pred_layer(ClassifierModule, 2048,
                                            [6, 12, 18, 24], [6, 12, 18, 24],
                                            num_classes)

        # for channels of four returned stages
        num_filters = [64, 128, 256, 512]
        self.feat_channels = [c * 4 for c in num_filters]

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False,
                    weight_attr=self.weight_attr_conv),
                nn.BatchNorm2D(
                    planes * block.expansion,
                    bias_attr=self.bias_attr,
                    weight_attr=self.weight_attr))
        nnlayers = nn.LayerList()
        nnlayers.append(
            block(
                self.inplanes,
                planes,
                stride,
                dilation=dilation,
                downsample=downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            nnlayers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*nnlayers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series,
                         num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        self.conv1_logit = x.clone()
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        # Resolution 1
        x_aug = self.layer5(x3)
        x_aug = F.interpolate(
            x_aug, size=input_size, mode='bilinear', align_corners=True)

        # Resolution 2
        x4 = self.layer4(x3)
        x6 = self.layer6(x4)
        x6 = F.interpolate(
            x6, size=input_size, mode='bilinear', align_corners=True)

        return [x6, x_aug, x1, x2, x3, x4]

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if not k.stop_gradient:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{
            'params': self.get_1x_lr_params_NOscale(),
            'lr': lr
        }, {
            'params': self.get_10x_lr_params(),
            'lr': 10 * lr
        }]


@manager.BACKBONES.add_component
def ResNet101(**args):
    model = ResNetMulti(
        Bottleneck, num_layers=[3, 4, 23, 3], **args)  # add pretrain
    return model

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

from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.functional import sigmoid

from paddleseg.cvlibs import manager, param_init
from paddleseg.utils import utils

from .rtformer import BasicBlock

BasicBlock = partial(BasicBlock, bias=False)
BasicBlock.expansion = 1

_interpolate = partial(F.interpolate, mode="bilinear", align_corners=False)


@manager.MODELS.add_component
class PIDNet(nn.Layer):
    def __init__(
            self,
            in_channels=3,
            m=2,
            n=3,
            num_classes=19,
            planes=64,
            ppm_planes=96,
            head_planes=128,
            augment=True,
            pretrained=None, ):
        super().__init__()
        self.augment = augment
        self.pretrained = pretrained

        # I Branch
        self.conv1 = nn.Sequential(
            nn.Conv2D(
                in_channels, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(planes),
            nn.ReLU(),
            nn.Conv2D(
                planes, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(planes),
            nn.ReLU(), )

        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(
            BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(
            BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(
            BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(
            Bottleneck, planes * 8, planes * 8, 2, stride=2)

        # P Branch
        self.compression3 = nn.Sequential(
            nn.Conv2D(
                planes * 4, planes * 2, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(planes * 2), )

        self.compression4 = nn.Sequential(
            nn.Conv2D(
                planes * 8, planes * 2, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(planes * 2), )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2,
                                                    planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2D(
                    planes * 4,
                    planes,
                    kernel_size=3,
                    padding=1,
                    bias_attr=False),
                nn.BatchNorm2D(planes), )
            self.diff4 = nn.Sequential(
                nn.Conv2D(
                    planes * 8,
                    planes * 2,
                    kernel_size=3,
                    padding=1,
                    bias_attr=False),
                nn.BatchNorm2D(planes * 2), )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2,
                                                    planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2,
                                                    planes * 2)
            self.diff3 = nn.Sequential(
                nn.Conv2D(
                    planes * 4,
                    planes * 2,
                    kernel_size=3,
                    padding=1,
                    bias_attr=False),
                nn.BatchNorm2D(planes * 2), )
            self.diff4 = nn.Sequential(
                nn.Conv2D(
                    planes * 8,
                    planes * 2,
                    kernel_size=3,
                    padding=1,
                    bias_attr=False),
                nn.BatchNorm2D(planes * 2), )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)

        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # Prediction Head
        if self.augment:
            self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
            self.seghead_d = SegmentHead(planes * 2, planes, 1)

        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)

        self.init_weight()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False, ),
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

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False, ),
                nn.BatchNorm2D(planes * block.expansion), )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)

        return layer

    def forward(self, x):
        h, w = x.shape[-2:]
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + _interpolate(
            self.diff3(x), size=[height_output, width_output])
        if self.augment:
            temp_p = x_

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + _interpolate(
            self.diff4(x), size=[height_output, width_output])
        if self.augment:
            temp_d = x_d

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = _interpolate(
            self.spp(self.layer5(x)), size=[height_output, width_output])
        x_ = self.final_layer(self.dfm(x_, x, x_d))
        if self.training or self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [
                F.interpolate(
                    x_extra_p, mode='bilinear', size=[h, w],
                    align_corners=True),
                F.interpolate(
                    x_, mode='bilinear', size=[h, w], align_corners=True),
                F.interpolate(
                    x_extra_d, mode='bilinear', size=[h, w],
                    align_corners=True),
            ]
        else:
            return [
                F.interpolate(
                    x_, mode='bilinear', size=[h, w], align_corners=True)
            ]

    def loss_computation(self, logits_list, losses, data):
        loss_list = []
        label = paddle.cast(data['label'], 'int32')
        for i in range(2):
            logits = logits_list[i]
            loss_i = losses['types'][i]
            coef_i = losses['coef'][i]
            loss_list.append(coef_i * loss_i(logits, label))
        s_loss = sum(loss_list)
        bd_loss = losses['coef'][2] * losses['types'][2](logits_list[2], data['edge'])
        filler = paddle.ones_like(label) * losses['types'][0].ignore_index
        bd_label = paddle.where(F.sigmoid(logits_list[-1][:, 0, :, :]) > 0.8, label, filler)
        sb_loss = losses['coef'][3] * losses['types'][3](logits_list[-2], bd_label)
        loss = bd_loss + sb_loss + s_loss
        return [loss.mean()]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
        else:
            for m in self.sublayers():
                if isinstance(m, nn.Conv2D):
                    param_init.kaiming_normal_init(m.weight)
                elif isinstance(m, nn.BatchNorm2D):
                    param_init.constant_init(m.weight, value=1.0)
                    param_init.constant_init(m.bias, value=0.0)


class Bottleneck(nn.Layer):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 no_relu=True):
        super().__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(
            planes, planes * self.expansion, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

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

        return out if self.no_relu else self.relu(out)


class SegmentHead(nn.Layer):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2D(inplanes)
        self.conv1 = nn.Conv2D(
            inplanes, interplanes, kernel_size=3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(interplanes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(interplanes, outplanes, kernel_size=1, padding=0)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = _interpolate(out, size=[height, width])
        return out


def _build_scale(kernel_size, stride, padding, in_chan, out_chan):
    return nn.Sequential(
        nn.AvgPool2D(
            kernel_size, stride, padding, exclusive=False),
        nn.BatchNorm2D(in_chan),
        nn.ReLU(),
        nn.Conv2D(
            in_chan, out_chan, 1, bias_attr=False), )


def _build_process(in_chan, out_chan, kernel_size=1, padding=0):
    return nn.Sequential(
        nn.BatchNorm2D(in_chan),
        nn.ReLU(),
        nn.Conv2D(
            in_chan, out_chan, kernel_size, padding=padding, bias_attr=False), )


class DAPPM(nn.Layer):
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        _build = partial(_build_scale, in_chan=inplanes, out_chan=branch_planes)
        self.scale1 = _build(5, 2, 2)
        self.scale2 = _build(9, 4, 4)
        self.scale3 = _build(17, 8, 8)
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.BatchNorm2D(inplanes),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )

        self.scale0 = _build_process(inplanes, branch_planes)

        self.process1 = _build_process(branch_planes, branch_planes, 3, 1)
        self.process2 = _build_process(branch_planes, branch_planes, 3, 1)
        self.process3 = _build_process(branch_planes, branch_planes, 3, 1)
        self.process4 = _build_process(branch_planes, branch_planes, 3, 1)
        self.compression = _build_process(branch_planes * 5, branch_planes, 3,
                                          1)
        self.shortcut = _build_process(inplanes, outplanes)

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(
            self.process1((_interpolate(
                self.scale1(x), size=[height, width]) + x_list[0])))
        x_list.append(
            self.process2((_interpolate(
                self.scale2(x), size=[height, width]) + x_list[1])))
        x_list.append(
            self.process3((_interpolate(
                self.scale3(x), size=[height, width]) + x_list[2])))
        x_list.append(
            self.process4((_interpolate(
                self.scale4(x), size=[height, width]) + x_list[3])))

        out = self.compression(paddle.concat(x_list, 1)) + self.shortcut(x)
        return out


class PAPPM(nn.Layer):
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        _build = partial(_build_scale, in_chan=inplanes, out_chan=branch_planes)
        self.scale1 = _build(5, 2, 2)
        self.scale2 = _build(9, 4, 4)
        self.scale3 = _build(17, 8, 8)
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.BatchNorm2D(inplanes),
            nn.ReLU(),
            nn.Conv2D(
                inplanes, branch_planes, kernel_size=1, bias_attr=False), )

        self.scale0 = _build_process(inplanes, branch_planes)

        self.scale_process = nn.Sequential(
            nn.BatchNorm2D(branch_planes * 4),
            nn.ReLU(),
            nn.Conv2D(
                branch_planes * 4,
                branch_planes * 4,
                kernel_size=3,
                padding=1,
                groups=4,
                bias_attr=False, ), )

        self.compression = _build_process(branch_planes * 5, outplanes)
        self.shortcut = _build_process(inplanes, outplanes)

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(
            _interpolate(
                self.scale1(x), size=[height, width]) + x_)
        scale_list.append(
            _interpolate(
                self.scale2(x), size=[height, width]) + x_)
        scale_list.append(
            _interpolate(
                self.scale3(x), size=[height, width]) + x_)
        scale_list.append(
            _interpolate(
                self.scale4(x), size=[height, width]) + x_)

        scale_out = self.scale_process(paddle.concat(scale_list, 1))
        out = self.compression(paddle.concat([x_, scale_out],
                                             1)) + self.shortcut(x)
        return out


def _build_convbn(in_chan, out_chan):
    return nn.Sequential(
        nn.Conv2D(
            in_chan, out_chan, kernel_size=1, bias_attr=False),
        nn.BatchNorm2D(out_chan), )


class PagFM(nn.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 after_relu=False,
                 with_channel=False):
        super().__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = _build_convbn(in_channels, mid_channels)
        self.f_y = _build_convbn(in_channels, mid_channels)
        if with_channel:
            self.up = _build_convbn(in_channels, mid_channels)

        if after_relu:
            self.relu = nn.ReLU()

    def forward(self, x, y):
        input_size = x.shape
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = _interpolate(y_q, size=[input_size[2], input_size[3]])
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = sigmoid(self.up(x_k * y_q))
        else:
            sim_map = sigmoid(
                paddle.unsqueeze(
                    paddle.sum(x_k * y_q, axis=1), 1))

        y = _interpolate(y, size=[input_size[2], input_size[3]])
        x = (1 - sim_map) * x + sim_map * y

        return x


class Light_Bag(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_p = _build_convbn(in_channels, out_channels)
        self.conv_i = _build_convbn(in_channels, out_channels)

    def forward(self, p, i, d):
        edge_att = sigmoid(d)

        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)

        return p_add + i_add


class DDFMv2(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_p = nn.Sequential(
            nn.BatchNorm2D(in_channels),
            nn.ReLU(),
            nn.Conv2D(
                in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2D(out_channels), )
        self.conv_i = nn.Sequential(
            nn.BatchNorm2D(in_channels),
            nn.ReLU(),
            nn.Conv2D(
                in_channels, out_channels, kernel_size=2),
            nn.BatchNorm2D(out_channels), )

    def forward(self, p, i, d):
        edge_att = sigmoid(d)

        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)

        return p_add + i_add


class Bag(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2D(in_channels),
            nn.ReLU(),
            nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias_attr=False), )

    def forward(self, p, i, d):
        edge_att = sigmoid(d)
        return self.conv(edge_att * p + (1 - edge_att) * i)

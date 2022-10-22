# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import math

import paddle
import paddle.nn as nn

from paddleseg.utils import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models.layers.layer_libs import SyncBatchNorm

__all__ = ["STDC1", "STDC2"]


class STDCNet(nn.Layer):
    """
    The STDCNet implementation based on PaddlePaddle.

    The original article refers to Meituan
    Fan, Mingyuan, et al. "Rethinking BiSeNet For Real-time Semantic Segmentation."
    (https://arxiv.org/abs/2104.13188)

    Args:
        base(int, optional): base channels. Default: 64.
        layers(list, optional): layers numbers list. It determines STDC block numbers of STDCNet's stage3\4\5. Defualt: [4, 5, 3].
        block_num(int,optional): block_num of features block. Default: 4.
        type(str,optional): feature fusion method "cat"/"add". Default: "cat".
        relative_lr(float,optional): parameters here receive a different learning rate when updating. The effective 
            learning rate is the prodcut of relative_lr and the global learning rate. Default: 1.0. 
        in_channels (int, optional): The channels of input image. Default: 3.
        pretrained(str, optional): the path of pretrained model.
    """

    def __init__(self,
                 base=64,
                 layers=[4, 5, 3],
                 block_num=4,
                 type="cat",
                 relative_lr=1.0,
                 in_channels=3,
                 pretrained=None):
        super(STDCNet, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.layers = layers
        self.feat_channels = [base // 2, base, base * 4, base * 8, base * 16]
        self.features = self._make_layers(in_channels, base, layers, block_num,
                                          block, relative_lr)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        """
        forward function for feature extract.
        """
        out_feats = []

        x = self.features[0](x)
        out_feats.append(x)
        x = self.features[1](x)
        out_feats.append(x)

        idx = [[2, 2 + self.layers[0]],
               [2 + self.layers[0], 2 + sum(self.layers[0:2])],
               [2 + sum(self.layers[0:2]), 2 + sum(self.layers)]]
        for start_idx, end_idx in idx:
            for i in range(start_idx, end_idx):
                x = self.features[i](x)
            out_feats.append(x)

        return out_feats

    def _make_layers(self, in_channels, base, layers, block_num, block,
                     relative_lr):
        features = []
        features += [ConvBNRelu(in_channels, base // 2, 3, 2, relative_lr)]
        features += [ConvBNRelu(base // 2, base, 3, 2, relative_lr)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(
                        block(base, base * 4, block_num, 2, relative_lr))
                elif j == 0:
                    features.append(
                        block(base * int(math.pow(2, i + 1)), base * int(
                            math.pow(2, i + 2)), block_num, 2, relative_lr))
                else:
                    features.append(
                        block(base * int(math.pow(2, i + 2)), base * int(
                            math.pow(2, i + 2)), block_num, 1, relative_lr))

        return nn.Sequential(*features)

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)


class ConvBNRelu(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel=3,
                 stride=1,
                 relative_lr=1.0):
        super(ConvBNRelu, self).__init__()
        param_attr = paddle.ParamAttr(learning_rate=relative_lr)
        self.conv = nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            weight_attr=param_attr,
            bias_attr=False)
        self.bn = nn.BatchNorm2D(
            out_planes, weight_attr=param_attr, bias_attr=param_attr)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 block_num=3,
                 stride=1,
                 relative_lr=1.0):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        param_attr = paddle.ParamAttr(learning_rate=relative_lr)
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    weight_attr=param_attr,
                    bias_attr=False),
                nn.BatchNorm2D(
                    out_planes // 2,
                    weight_attr=param_attr,
                    bias_attr=param_attr), )
            self.skip = nn.Sequential(
                nn.Conv2D(
                    in_planes,
                    in_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_planes,
                    weight_attr=param_attr,
                    bias_attr=False),
                nn.BatchNorm2D(
                    in_planes, weight_attr=param_attr, bias_attr=param_attr),
                nn.Conv2D(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    bias_attr=False,
                    weight_attr=param_attr),
                nn.BatchNorm2D(
                    out_planes, weight_attr=param_attr, bias_attr=param_attr), )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(
                        in_planes,
                        out_planes // 2,
                        kernel=1,
                        relative_lr=relative_lr))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 2,
                        stride=stride,
                        relative_lr=relative_lr))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 4,
                        stride=stride,
                        relative_lr=relative_lr))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx + 1)),
                        relative_lr=relative_lr))
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))),
                    relative_lr=relative_lr)

    def forward(self, x):
        out_list = []
        out = x
        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)
        if self.stride == 2:
            x = self.skip(x)
        return paddle.concat(out_list, axis=1) + x


class CatBottleneck(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 block_num=3,
                 stride=1,
                 relative_lr=1.0):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        param_attr = paddle.ParamAttr(learning_rate=relative_lr)
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    weight_attr=param_attr,
                    bias_attr=False),
                nn.BatchNorm2D(
                    out_planes // 2,
                    weight_attr=param_attr,
                    bias_attr=param_attr), )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(
                        in_planes,
                        out_planes // 2,
                        kernel=1,
                        relative_lr=relative_lr))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 2,
                        stride=stride,
                        relative_lr=relative_lr))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 4,
                        stride=stride,
                        relative_lr=relative_lr))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx + 1)),
                        relative_lr=relative_lr))
            else:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx)),
                        relative_lr=relative_lr))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        return out


@manager.BACKBONES.add_component
def STDC2(**kwargs):
    model = STDCNet(base=64, layers=[4, 5, 3], **kwargs)
    return model


@manager.BACKBONES.add_component
def STDC1(**kwargs):
    model = STDCNet(base=64, layers=[2, 2, 2], **kwargs)
    return model

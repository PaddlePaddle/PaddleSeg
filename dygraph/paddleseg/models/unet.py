# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2d
from paddle.nn import SyncBatchNorm as BatchNorm

from paddleseg.cvlibs import manager
from paddleseg import utils
from paddleseg.models.common import layer_libs


@manager.MODELS.add_component
class UNet(nn.Layer):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation.
    https://arxiv.org/abs/1505.04597

    Args:
        num_classes (int): the unique number of target classes.
        pretrained_model (str): the path of pretrained model.
        ignore_index (int): the value of ground-truth mask would be ignored while computing loss or doing evaluation. Default 255.
    """

    def __init__(self, num_classes, model_pretrained=None, ignore_index=255):
        super(UNet, self).__init__()
        self.model_pretrained = model_pretrained
        self.ignore_index = ignore_index

        self.encode = UnetEncoder()
        self.decode = UnetDecode()
        self.get_logit = GetLogit(64, num_classes)
        self.EPS = 1e-5

        self.init_weight()

    def forward(self, x, label=None):
        encode_data, short_cuts = self.encode(x)
        decode_data = self.decode(encode_data, short_cuts)
        logit = self.get_logit(decode_data)
        return [logit]

    def init_weight(self):
        """
        Initialize the parameters of model parts.
        """
        if self.model_pretrained is not None:
            if os.path.exists(self.model_pretrained):
                utils.load_pretrained_model(self, self.model_pretrained)
            else:
                raise Exception('Pretrained model is not found: {}'.format(
                    self.model_pretrained))


class UnetEncoder(nn.Layer):
    def __init__(self):
        super(UnetEncoder, self).__init__()
        self.double_conv = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        short_cuts.append(x)
        x = self.down1(x)
        short_cuts.append(x)
        x = self.down2(x)
        short_cuts.append(x)
        x = self.down3(x)
        short_cuts.append(x)
        x = self.down4(x)
        return x, short_cuts


class UnetDecode(nn.Layer):
    def __init__(self):
        super(UnetDecode, self).__init__()
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 64)

    def forward(self, x, short_cuts):
        x = self.up1(x, short_cuts[3])
        x = self.up2(x, short_cuts[2])
        x = self.up3(x, short_cuts[1])
        x = self.up4(x, short_cuts[0])
        return x


class DoubleConv(nn.Layer):
    def __init__(self, num_channels, num_filters):
        super(DoubleConv, self).__init__()
        self.conv0 = Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn0 = BatchNorm(num_filters)
        self.conv1 = Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn1 = BatchNorm(num_filters)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return x


class Down(nn.Layer):
    def __init__(self, num_channels, num_filters):
        super(Down, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(num_channels, num_filters)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.double_conv(x)
        return x


class Up(nn.Layer):
    def __init__(self, num_channels, num_filters):
        super(Up, self).__init__()
        self.double_conv = DoubleConv(2 * num_channels, num_filters)

    def forward(self, x, short_cut):
        x = F.resize_bilinear(x, short_cut.shape[2:])
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x


class GetLogit(nn.Layer):
    def __init__(self, num_channels, num_classes):
        super(GetLogit, self).__init__()
        self.conv = Conv2d(
            in_channels=num_channels,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

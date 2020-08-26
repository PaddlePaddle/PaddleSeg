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

import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D
from paddle.nn import SyncBatchNorm as BatchNorm

from dygraph.cvlibs import manager
from dygraph import utils


class UNet(fluid.dygraph.Layer):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation.
    https://arxiv.org/abs/1505.04597

    Args:
        num_classes (int): the unique number of target classes.
        pretrained_model (str): the path of pretrained model.
        ignore_index (int): the value of ground-truth mask would be ignored while computing loss or doing evaluation. Default 255.
    """

    def __init__(self, num_classes, pretrained_model=None, ignore_index=255):
        super(UNet, self).__init__()
        self.encode = UnetEncoder()
        self.decode = UnetDecode()
        self.get_logit = GetLogit(64, num_classes)
        self.ignore_index = ignore_index
        self.EPS = 1e-5

        self.init_weight(pretrained_model)

    def forward(self, x, label=None):
        encode_data, short_cuts = self.encode(x)
        decode_data = self.decode(encode_data, short_cuts)
        logit = self.get_logit(decode_data)
        if self.training:
            return self._get_loss(logit, label)
        else:
            score_map = fluid.layers.softmax(logit, axis=1)
            score_map = fluid.layers.transpose(score_map, [0, 2, 3, 1])
            pred = fluid.layers.argmax(score_map, axis=3)
            pred = fluid.layers.unsqueeze(pred, axes=[3])
            return pred, score_map

    def init_weight(self, pretrained_model=None):
        """
        Initialize the parameters of model parts.
        Args:
            pretrained_model ([str], optional): the pretrained_model path of backbone. Defaults to None.
        """
        if pretrained_model is not None:
            if os.path.exists(pretrained_model):
                utils.load_pretrained_model(self.backbone, pretrained_model)
                utils.load_pretrained_model(self, pretrained_model)
            else:
                raise Exception('Pretrained model is not found: {}'.format(
                    pretrained_model))

    def _get_loss(self, logit, label):
        logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
        label = fluid.layers.transpose(label, [0, 2, 3, 1])
        mask = label != self.ignore_index
        mask = fluid.layers.cast(mask, 'float32')
        loss, probs = fluid.layers.softmax_with_cross_entropy(
            logit,
            label,
            ignore_index=self.ignore_index,
            return_softmax=True,
            axis=-1)

        loss = loss * mask
        avg_loss = fluid.layers.mean(loss) / (
            fluid.layers.mean(mask) + self.EPS)

        label.stop_gradient = True
        mask.stop_gradient = True
        return avg_loss


class UnetEncoder(fluid.dygraph.Layer):
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


class UnetDecode(fluid.dygraph.Layer):
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


class DoubleConv(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters):
        super(DoubleConv, self).__init__()
        self.conv0 = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=1,
            padding=1)
        self.bn0 = BatchNorm(num_filters)
        self.conv1 = Conv2D(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=1,
            padding=1)
        self.bn1 = BatchNorm(num_filters)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = fluid.layers.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = fluid.layers.relu(x)
        return x


class Down(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters):
        super(Down, self).__init__()
        self.max_pool = Pool2D(
            pool_size=2, pool_type='max', pool_stride=2, pool_padding=0)
        self.double_conv = DoubleConv(num_channels, num_filters)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.double_conv(x)
        return x


class Up(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters):
        super(Up, self).__init__()
        self.double_conv = DoubleConv(2 * num_channels, num_filters)

    def forward(self, x, short_cut):
        short_cut_shape = fluid.layers.shape(short_cut)
        x = fluid.layers.resize_bilinear(x, short_cut_shape[2:])
        x = fluid.layers.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x


class GetLogit(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_classes):
        super(GetLogit, self).__init__()
        self.conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_classes,
            filter_size=3,
            stride=1,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


@manager.MODELS.add_component
def unet(*args, **kwargs):
    return UNet(*args, **kwargs)

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, BatchNorm, Pool2D
import contextlib

regularizer = fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0)
name_scope = ""


@contextlib.contextmanager
def scope(name):
    global name_scope
    bk = name_scope
    name_scope = name_scope + name + '/'
    yield
    name_scope = bk


class UNet(fluid.dygraph.Layer):
    def __init__(self, num_classes, upsample_mode='bilinear', ignore_index=255):
        super().__init__()
        self.encode = Encoder()
        self.decode = Decode(upsample_mode=upsample_mode)
        self.get_logit = GetLogit(64, num_classes)
        self.ignore_index = ignore_index

    def forward(self, x, label=None, mode='train'):
        encode_data, short_cuts = self.encode(x)
        decode_data = self.decode(encode_data, short_cuts)
        logit = self.get_logit(decode_data)
        if mode == 'train':
            return self._get_loss(logit, label)
        else:
            score_map = fluid.layers.softmax(logit, axis=1)
            score_map = fluid.layers.transpose(score_map, [0, 2, 3, 1])
            pred = fluid.layers.argmax(score_map, axis=3)
            pred = fluid.layers.unsqueeze(pred, axes=[3])
            return pred, score_map

    def _get_loss(self, logit, label):
        mask = label != self.ignore_index
        mask = fluid.layers.cast(mask, 'float32')
        loss, probs = fluid.layers.softmax_with_cross_entropy(
            logit,
            label,
            ignore_index=self.ignore_index,
            return_softmax=True,
            axis=1)

        loss = loss * mask
        avg_loss = fluid.layers.mean(loss) / (fluid.layers.mean(mask) + 0.00001)

        label.stop_gradient = True
        mask.stop_gradient = True
        return avg_loss


class Encoder(fluid.dygraph.Layer):
    def __init__(self):
        super().__init__()
        with scope('encode'):
            with scope('block1'):
                self.double_conv = DoubleConv(3, 64)
            with scope('block1'):
                self.down1 = Down(64, 128)
            with scope('block2'):
                self.down2 = Down(128, 256)
            with scope('block3'):
                self.down3 = Down(256, 512)
            with scope('block4'):
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


class Decode(fluid.dygraph.Layer):
    def __init__(self, upsample_mode='bilinear'):
        super().__init__()
        with scope('decode'):
            with scope('decode1'):
                self.up1 = Up(512, 256, upsample_mode)
            with scope('decode2'):
                self.up2 = Up(256, 128, upsample_mode)
            with scope('decode3'):
                self.up3 = Up(128, 64, upsample_mode)
            with scope('decode4'):
                self.up4 = Up(64, 64, upsample_mode)

    def forward(self, x, short_cuts):
        x = self.up1(x, short_cuts[3])
        x = self.up2(x, short_cuts[2])
        x = self.up3(x, short_cuts[1])
        x = self.up4(x, short_cuts[0])
        return x


class GetLogit(fluid.dygraph.Layer):
    def __init__(self):
        super().__init__()


class DoubleConv(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters):
        super().__init__()
        with scope('conv0'):
            param_attr = fluid.ParamAttr(
                name=name_scope + 'weights',
                regularizer=regularizer,
                initializer=fluid.initializer.TruncatedNormal(
                    loc=0.0, scale=0.33))
            self.conv0 = Conv2D(
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=3,
                stride=1,
                padding=1,
                param_attr=param_attr)
            self.bn0 = BatchNorm(
                num_channels=num_filters,
                param_attr=fluid.ParamAttr(
                    name=name_scope + 'gamma', regularizer=regularizer),
                bias_attr=fluid.ParamAttr(
                    name=name_scope + 'beta', regularizer=regularizer),
                moving_mean_name=name_scope + 'moving_mean',
                moving_variance_name=name_scope + 'moving_variance')
        with scope('conv1'):
            param_attr = fluid.ParamAttr(
                name=name_scope + 'weights',
                regularizer=regularizer,
                initializer=fluid.initializer.TruncatedNormal(
                    loc=0.0, scale=0.33))
            self.conv1 = Conv2D(
                num_channels=num_filters,
                num_filters=num_filters,
                filter_size=3,
                stride=1,
                padding=1,
                param_attr=param_attr)
            self.bn1 = BatchNorm(
                num_channels=num_filters,
                param_attr=fluid.ParamAttr(
                    name=name_scope + 'gamma', regularizer=regularizer),
                bias_attr=fluid.ParamAttr(
                    name=name_scope + 'beta', regularizer=regularizer),
                moving_mean_name=name_scope + 'moving_mean',
                moving_variance_name=name_scope + 'moving_variance')

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
        super().__init__()
        with scope("down"):
            self.max_pool = Pool2D(
                pool_size=2, pool_type='max', pool_stride=2, pool_padding=0)
            self.double_conv = DoubleConv(num_channels, num_filters)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.double_conv(x)
        return x


class Up(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, upsample_mode):
        super().__init__()
        self.upsample_mode = upsample_mode
        with scope('up'):
            if upsample_mode == 'bilinear':
                self.double_conv = DoubleConv(2 * num_channels, num_filters)
            if not upsample_mode == 'bilinear':
                param_attr = fluid.ParamAttr(
                    name=name_scope + 'weights',
                    regularizer=regularizer,
                    initializer=fluid.initializer.XavierInitializer(),
                )
                self.deconv = fluid.dygraph.Conv2DTranspose(
                    num_channels=num_channels,
                    num_filters=num_filters // 2,
                    filter_size=2,
                    stride=2,
                    padding=0,
                    param_attr=param_attr)
                self.double_conv = DoubleConv(num_channels + num_filters // 2,
                                              num_filters)

    def forward(self, x, short_cut):
        if self.upsample_mode == 'bilinear':
            short_cut_shape = fluid.layers.shape(short_cut)
            x = fluid.layers.resize_bilinear(x, short_cut_shape[2:])
        else:
            x = self.deconv(x)
        x = fluid.layers.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x


class GetLogit(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        with scope('logit'):
            param_attr = fluid.ParamAttr(
                name=name_scope + 'weights',
                regularizer=regularizer,
                initializer=fluid.initializer.TruncatedNormal(
                    loc=0.0, scale=0.01))
            self.conv = Conv2D(
                num_channels=num_channels,
                num_filters=num_classes,
                filter_size=3,
                stride=1,
                padding=1,
                param_attr=param_attr)

    def forward(self, x):
        x = self.conv(x)
        return x

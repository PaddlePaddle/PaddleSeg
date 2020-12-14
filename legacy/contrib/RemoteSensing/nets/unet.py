# coding: utf8
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
from .libs import scope, name_scope
from .libs import bn, bn_relu, relu
from .libs import conv, max_pool, deconv
from .libs import sigmoid_to_softmax
from .loss import softmax_with_loss
from .loss import dice_loss
from .loss import bce_loss


class UNet(object):
    """实现Unet模型
        `"U-Net: Convolutional Networks for Biomedical Image Segmentation"
        <https://arxiv.org/abs/1505.04597>`

        Args:
            num_classes (int): 类别数
            mode (str): 网络运行模式，根据mode构建网络的输入和返回。
                当mode为'train'时，输入为image(-1, 3, -1, -1)和label (-1, 1, -1, -1) 返回loss。
                当mode为'train'时，输入为image (-1, 3, -1, -1)和label  (-1, 1, -1, -1)，返回loss，
                pred (与网络输入label 相同大小的预测结果，值代表相应的类别），label，mask（非忽略值的mask，
                与label相同大小，bool类型）。
                当mode为'test'时，输入为image(-1, 3, -1, -1)返回pred (-1, 1, -1, -1)和
                logit (-1, num_classes, -1, -1) 通道维上代表每一类的概率值。
            upsample_mode (str): UNet decode时采用的上采样方式，取值为'bilinear'时利用双线行差值进行上菜样，
                当输入其他选项时则利用反卷积进行上菜样，默认为'bilinear'。
            use_bce_loss (bool): 是否使用bce loss作为网络的损失函数，只能用于两类分割。可与dice loss同时使用。
            use_dice_loss (bool): 是否使用dice loss作为网络的损失函数，只能用于两类分割，可与bce loss同时使用。
                当use_bce_loss和use_dice_loss都为False时，使用交叉熵损失函数。
            class_weight (list/str): 交叉熵损失函数各类损失的权重。当class_weight为list的时候，长度应为
                num_classes。当class_weight为str时， weight.lower()应为'dynamic'，这时会根据每一轮各类像素的比重
                自行计算相应的权重，每一类的权重为：每类的比例 * num_classes。class_weight取默认值None是，各类的权重1，
                即平时使用的交叉熵损失函数。
            ignore_index (int): label上忽略的值，label为ignore_index的像素不参与损失函数的计算。

        Raises:
            ValueError: use_bce_loss或use_dice_loss为真且num_calsses > 2。
            ValueError: class_weight为list, 但长度不等于num_class。
                class_weight为str, 但class_weight.low()不等于dynamic。
            TypeError: class_weight不为None时，其类型不是list或str。
        """

    def __init__(self,
                 num_classes,
                 mode='train',
                 upsample_mode='bilinear',
                 input_channel=3,
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=255):
        # dice_loss或bce_loss只适用两类分割中
        if num_classes > 2 and (use_bce_loss or use_dice_loss):
            raise Exception(
                "dice loss and bce loss is only applicable to binary classfication"
            )

        if class_weight is not None:
            if isinstance(class_weight, list):
                if len(class_weight) != num_classes:
                    raise ValueError(
                        "Length of class_weight should be equal to number of classes"
                    )
            elif isinstance(class_weight, str):
                if class_weight.lower() != 'dynamic':
                    raise ValueError(
                        "if class_weight is string, must be dynamic!")
            else:
                raise TypeError(
                    'Expect class_weight is a list or string but receive {}'.
                    format(type(class_weight)))
        self.num_classes = num_classes
        self.mode = mode
        self.upsample_mode = upsample_mode
        self.input_channel = input_channel
        self.use_bce_loss = use_bce_loss
        self.use_dice_loss = use_dice_loss
        self.class_weight = class_weight
        self.ignore_index = ignore_index

    def _double_conv(self, data, out_ch):
        param_attr = fluid.ParamAttr(
            name='weights',
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=0.0),
            initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.33))
        with scope("conv0"):
            data = bn_relu(
                conv(
                    data, out_ch, 3, stride=1, padding=1,
                    param_attr=param_attr))
        with scope("conv1"):
            data = bn_relu(
                conv(
                    data, out_ch, 3, stride=1, padding=1,
                    param_attr=param_attr))
        return data

    def _down(self, data, out_ch):
        # 下采样：max_pool + 2个卷积
        with scope("down"):
            data = max_pool(data, 2, 2, 0)
            data = self._double_conv(data, out_ch)
        return data

    def _up(self, data, short_cut, out_ch):
        # 上采样：data上采样(resize或deconv), 并与short_cut concat
        param_attr = fluid.ParamAttr(
            name='weights',
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=0.0),
            initializer=fluid.initializer.XavierInitializer(),
        )
        with scope("up"):
            if self.upsample_mode == 'bilinear':
                short_cut_shape = fluid.layers.shape(short_cut)
                data = fluid.layers.resize_bilinear(data, short_cut_shape[2:])
            else:
                data = deconv(
                    data,
                    out_ch // 2,
                    filter_size=2,
                    stride=2,
                    padding=0,
                    param_attr=param_attr)
            data = fluid.layers.concat([data, short_cut], axis=1)
            data = self._double_conv(data, out_ch)
        return data

    def _encode(self, data):
        # 编码器设置
        short_cuts = []
        with scope("encode"):
            with scope("block1"):
                data = self._double_conv(data, 64)
                short_cuts.append(data)
            with scope("block2"):
                data = self._down(data, 128)
                short_cuts.append(data)
            with scope("block3"):
                data = self._down(data, 256)
                short_cuts.append(data)
            with scope("block4"):
                data = self._down(data, 512)
                short_cuts.append(data)
            with scope("block5"):
                data = self._down(data, 512)
        return data, short_cuts

    def _decode(self, data, short_cuts):
        # 解码器设置，与编码器对称
        with scope("decode"):
            with scope("decode1"):
                data = self._up(data, short_cuts[3], 256)
            with scope("decode2"):
                data = self._up(data, short_cuts[2], 128)
            with scope("decode3"):
                data = self._up(data, short_cuts[1], 64)
            with scope("decode4"):
                data = self._up(data, short_cuts[0], 64)
        return data

    def _get_logit(self, data, num_classes):
        # 根据类别数设置最后一个卷积层输出
        param_attr = fluid.ParamAttr(
            name='weights',
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=0.0),
            initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01))
        with scope("logit"):
            data = conv(
                data,
                num_classes,
                3,
                stride=1,
                padding=1,
                param_attr=param_attr)
        return data

    def _get_loss(self, logit, label, mask):
        avg_loss = 0
        if not (self.use_dice_loss or self.use_bce_loss):
            avg_loss += softmax_with_loss(
                logit,
                label,
                mask,
                num_classes=self.num_classes,
                weight=self.class_weight,
                ignore_index=self.ignore_index)
        else:
            if self.use_dice_loss:
                avg_loss += dice_loss(logit, label, mask)
            if self.use_bce_loss:
                avg_loss += bce_loss(
                    logit, label, mask, ignore_index=self.ignore_index)

        return avg_loss

    def generate_inputs(self):
        inputs = OrderedDict()
        inputs['image'] = fluid.data(
            dtype='float32',
            shape=[None, self.input_channel, None, None],
            name='image')
        if self.mode == 'train':
            inputs['label'] = fluid.data(
                dtype='int32', shape=[None, 1, None, None], name='label')
        elif self.mode == 'eval':
            inputs['label'] = fluid.data(
                dtype='int32', shape=[None, 1, None, None], name='label')
        return inputs

    def build_net(self, inputs):
        # 在两类分割情况下，当loss函数选择dice_loss或bce_loss的时候，最后logit输出通道数设置为1
        if self.use_dice_loss or self.use_bce_loss:
            self.num_classes = 1

        image = inputs['image']
        encode_data, short_cuts = self._encode(image)
        decode_data = self._decode(encode_data, short_cuts)
        logit = self._get_logit(decode_data, self.num_classes)

        if self.num_classes == 1:
            out = sigmoid_to_softmax(logit)
            out = fluid.layers.transpose(out, [0, 2, 3, 1])
        else:
            out = fluid.layers.transpose(logit, [0, 2, 3, 1])

        pred = fluid.layers.argmax(out, axis=3)
        pred = fluid.layers.unsqueeze(pred, axes=[3])

        if self.mode == 'train':
            label = inputs['label']
            mask = label != self.ignore_index
            return self._get_loss(logit, label, mask)

        elif self.mode == 'eval':
            label = inputs['label']
            mask = label != self.ignore_index
            loss = self._get_loss(logit, label, mask)
            return loss, pred, label, mask
        else:
            if self.num_classes == 1:
                logit = sigmoid_to_softmax(logit)
            else:
                logit = fluid.layers.softmax(logit, axis=1)
            return pred, logit

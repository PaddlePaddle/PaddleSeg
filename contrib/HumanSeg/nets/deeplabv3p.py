# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
from .libs import bn_relu, relu
from .libs import conv
from .libs import separate_conv
from .libs import sigmoid_to_softmax
from .seg_modules import softmax_with_loss
from .seg_modules import dice_loss
from .seg_modules import bce_loss
from .backbone import MobileNetV2
from .backbone import Xception


class DeepLabv3p(object):
    """实现DeepLabv3+模型
    `"Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1802.02611>`

    Args:
        num_classes (int): 类别数。
        backbone (str): DeepLabv3+的backbone网络，实现特征图的计算，取值范围为['Xception65', 'Xception41',
            'MobileNetV2_x0.25', 'MobileNetV2_x0.5', 'MobileNetV2_x1.0', 'MobileNetV2_x1.5',
            'MobileNetV2_x2.0']。默认'MobileNetV2_x1.0'。
        mode (str): 网络运行模式，根据mode构建网络的输入和返回。
            当mode为'train'时，输入为image(-1, 3, -1, -1)和label (-1, 1, -1, -1) 返回loss。
            当mode为'train'时，输入为image (-1, 3, -1, -1)和label  (-1, 1, -1, -1)，返回loss，
            pred (与网络输入label 相同大小的预测结果，值代表相应的类别），label，mask（非忽略值的mask，
            与label相同大小，bool类型）。
            当mode为'test'时，输入为image(-1, 3, -1, -1)返回pred (-1, 1, -1, -1)和
            logit (-1, num_classes, -1, -1) 通道维上代表每一类的概率值。
        output_stride (int): backbone 输出特征图相对于输入的下采样倍数，一般取值为8或16。
        aspp_with_sep_conv (bool): 在asspp模块是否采用separable convolutions。
        decoder_use_sep_conv (bool)： decoder模块是否采用separable convolutions。
        encoder_with_aspp (bool): 是否在encoder阶段采用aspp模块。
        enable_decoder (bool): 是否使用decoder模块。
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
                 backbone='MobileNetV2_x1.0',
                 mode='train',
                 output_stride=16,
                 aspp_with_sep_conv=True,
                 decoder_use_sep_conv=True,
                 encoder_with_aspp=True,
                 enable_decoder=True,
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=255):
        # dice_loss或bce_loss只适用两类分割中
        if num_classes > 2 and (use_bce_loss or use_dice_loss):
            raise ValueError(
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
        self.backbone = backbone
        self.mode = mode
        self.use_bce_loss = use_bce_loss
        self.use_dice_loss = use_dice_loss
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.output_stride = output_stride
        self.aspp_with_sep_conv = aspp_with_sep_conv
        self.decoder_use_sep_conv = decoder_use_sep_conv
        self.encoder_with_aspp = encoder_with_aspp
        self.enable_decoder = enable_decoder

    def _get_backbone(self, backbone):
        def mobilenetv2(backbone):
            # backbone: xception结构配置
            # output_stride：下采样倍数
            # end_points: mobilenetv2的block数
            # decode_point: 从mobilenetv2中引出分支所在block数, 作为decoder输入
            if '0.25' in backbone:
                scale = 0.25
            elif '0.5' in backbone:
                scale = 0.5
            elif '1.0' in backbone:
                scale = 1.0
            elif '1.5' in backbone:
                scale = 1.5
            elif '2.0' in backbone:
                scale = 2.0
            end_points = 18
            decode_points = 4
            return MobileNetV2(
                scale=scale,
                output_stride=self.output_stride,
                end_points=end_points,
                decode_points=decode_points)

        def xception(backbone):
            # decode_point: 从Xception中引出分支所在block数，作为decoder输入
            # end_point：Xception的block数
            if '65' in backbone:
                decode_points = 2
                end_points = 21
                layers = 65
            if '41' in backbone:
                decode_points = 2
                end_points = 13
                layers = 41
            if '71' in backbone:
                decode_points = 3
                end_points = 23
                layers = 71
            return Xception(
                layers=layers,
                output_stride=self.output_stride,
                end_points=end_points,
                decode_points=decode_points)

        if 'Xception' in backbone:
            return xception(backbone)
        elif 'MobileNetV2' in backbone:
            return mobilenetv2(backbone)

    def _encoder(self, input):
        # 编码器配置，采用ASPP架构，pooling + 1x1_conv + 三个不同尺度的空洞卷积并行, concat后1x1conv
        # ASPP_WITH_SEP_CONV：默认为真，使用depthwise可分离卷积，否则使用普通卷积
        # OUTPUT_STRIDE: 下采样倍数，8或16，决定aspp_ratios大小
        # aspp_ratios：ASPP模块空洞卷积的采样率

        if self.output_stride == 16:
            aspp_ratios = [6, 12, 18]
        elif self.output_stride == 8:
            aspp_ratios = [12, 24, 36]
        else:
            raise Exception("DeepLabv3p only support stride 8 or 16")

        param_attr = fluid.ParamAttr(
            name=name_scope + 'weights',
            regularizer=None,
            initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.06))
        with scope('encoder'):
            channel = 256
            with scope("image_pool"):
                image_avg = fluid.layers.reduce_mean(
                    input, [2, 3], keep_dim=True)
                image_avg = bn_relu(
                    conv(
                        image_avg,
                        channel,
                        1,
                        1,
                        groups=1,
                        padding=0,
                        param_attr=param_attr))
                input_shape = fluid.layers.shape(input)
                image_avg = fluid.layers.resize_bilinear(
                    image_avg, input_shape[2:])

            with scope("aspp0"):
                aspp0 = bn_relu(
                    conv(
                        input,
                        channel,
                        1,
                        1,
                        groups=1,
                        padding=0,
                        param_attr=param_attr))
            with scope("aspp1"):
                if self.aspp_with_sep_conv:
                    aspp1 = separate_conv(
                        input, channel, 1, 3, dilation=aspp_ratios[0], act=relu)
                else:
                    aspp1 = bn_relu(
                        conv(
                            input,
                            channel,
                            stride=1,
                            filter_size=3,
                            dilation=aspp_ratios[0],
                            padding=aspp_ratios[0],
                            param_attr=param_attr))
            with scope("aspp2"):
                if self.aspp_with_sep_conv:
                    aspp2 = separate_conv(
                        input, channel, 1, 3, dilation=aspp_ratios[1], act=relu)
                else:
                    aspp2 = bn_relu(
                        conv(
                            input,
                            channel,
                            stride=1,
                            filter_size=3,
                            dilation=aspp_ratios[1],
                            padding=aspp_ratios[1],
                            param_attr=param_attr))
            with scope("aspp3"):
                if self.aspp_with_sep_conv:
                    aspp3 = separate_conv(
                        input, channel, 1, 3, dilation=aspp_ratios[2], act=relu)
                else:
                    aspp3 = bn_relu(
                        conv(
                            input,
                            channel,
                            stride=1,
                            filter_size=3,
                            dilation=aspp_ratios[2],
                            padding=aspp_ratios[2],
                            param_attr=param_attr))
            with scope("concat"):
                data = fluid.layers.concat(
                    [image_avg, aspp0, aspp1, aspp2, aspp3], axis=1)
                data = bn_relu(
                    conv(
                        data,
                        channel,
                        1,
                        1,
                        groups=1,
                        padding=0,
                        param_attr=param_attr))
                data = fluid.layers.dropout(data, 0.9)
            return data

    def _decoder(self, encode_data, decode_shortcut):
        # 解码器配置
        # encode_data：编码器输出
        # decode_shortcut: 从backbone引出的分支, resize后与encode_data concat
        # decoder_use_sep_conv: 默认为真，则concat后连接两个可分离卷积，否则为普通卷积
        param_attr = fluid.ParamAttr(
            name=name_scope + 'weights',
            regularizer=None,
            initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.06))
        with scope('decoder'):
            with scope('concat'):
                decode_shortcut = bn_relu(
                    conv(
                        decode_shortcut,
                        48,
                        1,
                        1,
                        groups=1,
                        padding=0,
                        param_attr=param_attr))

                decode_shortcut_shape = fluid.layers.shape(decode_shortcut)
                encode_data = fluid.layers.resize_bilinear(
                    encode_data, decode_shortcut_shape[2:])
                encode_data = fluid.layers.concat(
                    [encode_data, decode_shortcut], axis=1)
            if self.decoder_use_sep_conv:
                with scope("separable_conv1"):
                    encode_data = separate_conv(
                        encode_data, 256, 1, 3, dilation=1, act=relu)
                with scope("separable_conv2"):
                    encode_data = separate_conv(
                        encode_data, 256, 1, 3, dilation=1, act=relu)
            else:
                with scope("decoder_conv1"):
                    encode_data = bn_relu(
                        conv(
                            encode_data,
                            256,
                            stride=1,
                            filter_size=3,
                            dilation=1,
                            padding=1,
                            param_attr=param_attr))
                with scope("decoder_conv2"):
                    encode_data = bn_relu(
                        conv(
                            encode_data,
                            256,
                            stride=1,
                            filter_size=3,
                            dilation=1,
                            padding=1,
                            param_attr=param_attr))
            return encode_data

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
            dtype='float32', shape=[None, 3, None, None], name='image')
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

        backbone_net = self._get_backbone(self.backbone)
        data, decode_shortcuts = backbone_net(image)
        decode_shortcut = decode_shortcuts[backbone_net.decode_points]

        # 编码器解码器设置
        if self.encoder_with_aspp:
            data = self._encoder(data)
        if self.enable_decoder:
            data = self._decoder(data, decode_shortcut)

        # 根据类别数设置最后一个卷积层输出，并resize到图片原始尺寸
        param_attr = fluid.ParamAttr(
            name=name_scope + 'weights',
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=0.0),
            initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01))
        with scope('logit'):
            with fluid.name_scope('last_conv'):
                logit = conv(
                    data,
                    self.num_classes,
                    1,
                    stride=1,
                    padding=0,
                    bias_attr=True,
                    param_attr=param_attr)
            image_shape = fluid.layers.shape(image)
            logit = fluid.layers.resize_bilinear(logit, image_shape[2:])

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

        else:
            if self.num_classes == 1:
                logit = sigmoid_to_softmax(logit)
            else:
                logit = fluid.layers.softmax(logit, axis=1)
            return pred, logit

        return logit

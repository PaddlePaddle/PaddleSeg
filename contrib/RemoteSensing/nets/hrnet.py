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
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from .loss import softmax_with_loss
from .loss import dice_loss
from .loss import bce_loss
from .libs import sigmoid_to_softmax


class HRNet(object):
    def __init__(self,
                 num_classes,
                 input_channel=3,
                 mode='train',
                 stage1_num_modules=1,
                 stage1_num_blocks=[4],
                 stage1_num_channels=[64],
                 stage2_num_modules=1,
                 stage2_num_blocks=[4, 4],
                 stage2_num_channels=[18, 36],
                 stage3_num_modules=4,
                 stage3_num_blocks=[4, 4, 4],
                 stage3_num_channels=[18, 36, 72],
                 stage4_num_modules=3,
                 stage4_num_blocks=[4, 4, 4, 4],
                 stage4_num_channels=[18, 36, 72, 144],
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
        self.input_channel = input_channel
        self.mode = mode
        self.use_bce_loss = use_bce_loss
        self.use_dice_loss = use_dice_loss
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.stage1_num_modules = stage1_num_modules
        self.stage1_num_blocks = stage1_num_blocks
        self.stage1_num_channels = stage1_num_channels
        self.stage2_num_modules = stage2_num_modules
        self.stage2_num_blocks = stage2_num_blocks
        self.stage2_num_channels = stage2_num_channels
        self.stage3_num_modules = stage3_num_modules
        self.stage3_num_blocks = stage3_num_blocks
        self.stage3_num_channels = stage3_num_channels
        self.stage4_num_modules = stage4_num_modules
        self.stage4_num_blocks = stage4_num_blocks
        self.stage4_num_channels = stage4_num_channels

    def build_net(self, inputs):
        if self.use_dice_loss or self.use_bce_loss:
            self.num_classes = 1
        image = inputs['image']
        logit = self._high_resolution_net(image, self.num_classes)
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

    def _conv_bn_layer(self,
                       input,
                       filter_size,
                       num_filters,
                       stride=1,
                       padding=1,
                       num_groups=1,
                       if_act=True,
                       name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            act=None,
            param_attr=ParamAttr(initializer=MSRA(), name=name + '_weights'),
            bias_attr=False)
        bn_name = name + '_bn'
        bn = fluid.layers.batch_norm(
            input=conv,
            param_attr=ParamAttr(
                name=bn_name + "_scale",
                initializer=fluid.initializer.Constant(1.0)),
            bias_attr=ParamAttr(
                name=bn_name + "_offset",
                initializer=fluid.initializer.Constant(0.0)),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
        if if_act:
            bn = fluid.layers.relu(bn)
        return bn

    def _basic_block(self,
                     input,
                     num_filters,
                     stride=1,
                     downsample=False,
                     name=None):
        residual = input
        conv = self._conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=num_filters,
            stride=stride,
            name=name + '_conv1')
        conv = self._conv_bn_layer(
            input=conv,
            filter_size=3,
            num_filters=num_filters,
            if_act=False,
            name=name + '_conv2')
        if downsample:
            residual = self._conv_bn_layer(
                input=input,
                filter_size=1,
                num_filters=num_filters,
                if_act=False,
                name=name + '_downsample')
        return fluid.layers.elementwise_add(x=residual, y=conv, act='relu')

    def _bottleneck_block(self,
                          input,
                          num_filters,
                          stride=1,
                          downsample=False,
                          name=None):
        residual = input
        conv = self._conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=num_filters,
            name=name + '_conv1')
        conv = self._conv_bn_layer(
            input=conv,
            filter_size=3,
            num_filters=num_filters,
            stride=stride,
            name=name + '_conv2')
        conv = self._conv_bn_layer(
            input=conv,
            filter_size=1,
            num_filters=num_filters * 4,
            if_act=False,
            name=name + '_conv3')
        if downsample:
            residual = self._conv_bn_layer(
                input=input,
                filter_size=1,
                num_filters=num_filters * 4,
                if_act=False,
                name=name + '_downsample')
        return fluid.layers.elementwise_add(x=residual, y=conv, act='relu')

    def _fuse_layers(self, x, channels, multi_scale_output=True, name=None):
        out = []
        for i in range(len(channels) if multi_scale_output else 1):
            residual = x[i]
            shape = fluid.layers.shape(residual)[-2:]
            for j in range(len(channels)):
                if j > i:
                    y = self._conv_bn_layer(
                        x[j],
                        filter_size=1,
                        num_filters=channels[i],
                        if_act=False,
                        name=name + '_layer_' + str(i + 1) + '_' + str(j + 1))
                    y = fluid.layers.resize_bilinear(input=y, out_shape=shape)
                    residual = fluid.layers.elementwise_add(
                        x=residual, y=y, act=None)
                elif j < i:
                    y = x[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            y = self._conv_bn_layer(
                                y,
                                filter_size=3,
                                num_filters=channels[i],
                                stride=2,
                                if_act=False,
                                name=name + '_layer_' + str(i + 1) + '_' +
                                str(j + 1) + '_' + str(k + 1))
                        else:
                            y = self._conv_bn_layer(
                                y,
                                filter_size=3,
                                num_filters=channels[j],
                                stride=2,
                                name=name + '_layer_' + str(i + 1) + '_' +
                                str(j + 1) + '_' + str(k + 1))
                    residual = fluid.layers.elementwise_add(
                        x=residual, y=y, act=None)

            residual = fluid.layers.relu(residual)
            out.append(residual)
        return out

    def _branches(self, x, block_num, channels, name=None):
        out = []
        for i in range(len(channels)):
            residual = x[i]
            for j in range(block_num[i]):
                residual = self._basic_block(
                    residual,
                    channels[i],
                    name=name + '_branch_layer_' + str(i + 1) + '_' +
                    str(j + 1))
            out.append(residual)
        return out

    def _high_resolution_module(self,
                                x,
                                blocks,
                                channels,
                                multi_scale_output=True,
                                name=None):
        residual = self._branches(x, blocks, channels, name=name)
        out = self._fuse_layers(
            residual,
            channels,
            multi_scale_output=multi_scale_output,
            name=name)
        return out

    def _transition_layer(self, x, in_channels, out_channels, name=None):
        num_in = len(in_channels)
        num_out = len(out_channels)
        out = []
        for i in range(num_out):
            if i < num_in:
                if in_channels[i] != out_channels[i]:
                    residual = self._conv_bn_layer(
                        x[i],
                        filter_size=3,
                        num_filters=out_channels[i],
                        name=name + '_layer_' + str(i + 1))
                    out.append(residual)
                else:
                    out.append(x[i])
            else:
                residual = self._conv_bn_layer(
                    x[-1],
                    filter_size=3,
                    num_filters=out_channels[i],
                    stride=2,
                    name=name + '_layer_' + str(i + 1))
                out.append(residual)
        return out

    def _stage(self,
               x,
               num_modules,
               num_blocks,
               num_channels,
               multi_scale_output=True,
               name=None):
        out = x
        for i in range(num_modules):
            if i == num_modules - 1 and multi_scale_output == False:
                out = self._high_resolution_module(
                    out,
                    num_blocks,
                    num_channels,
                    multi_scale_output=False,
                    name=name + '_' + str(i + 1))
            else:
                out = self._high_resolution_module(
                    out, num_blocks, num_channels, name=name + '_' + str(i + 1))

        return out

    def _layer1(self, input, num_modules, num_blocks, num_channels, name=None):
        # num_modules 默认为1,是否增加处理，官网实现为[1]，是否对齐。
        conv = input
        for i in range(num_blocks[0]):
            conv = self._bottleneck_block(
                conv,
                num_filters=num_channels[0],
                downsample=True if i == 0 else False,
                name=name + '_' + str(i + 1))
        return conv

    def _high_resolution_net(self, input, num_classes):
        x = self._conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=self.stage1_num_channels[0],
            stride=2,
            if_act=True,
            name='layer1_1')
        x = self._conv_bn_layer(
            input=x,
            filter_size=3,
            num_filters=self.stage1_num_channels[0],
            stride=2,
            if_act=True,
            name='layer1_2')

        la1 = self._layer1(
            x,
            self.stage1_num_modules,
            self.stage1_num_blocks,
            self.stage1_num_channels,
            name='layer2')
        tr1 = self._transition_layer([la1],
                                     self.stage1_num_channels,
                                     self.stage2_num_channels,
                                     name='tr1')
        st2 = self._stage(
            tr1,
            self.stage2_num_modules,
            self.stage2_num_blocks,
            self.stage2_num_channels,
            name='st2')
        tr2 = self._transition_layer(
            st2, self.stage2_num_channels, self.stage3_num_channels, name='tr2')
        st3 = self._stage(
            tr2,
            self.stage3_num_modules,
            self.stage3_num_blocks,
            self.stage3_num_channels,
            name='st3')
        tr3 = self._transition_layer(
            st3, self.stage3_num_channels, self.stage4_num_channels, name='tr3')
        st4 = self._stage(
            tr3,
            self.stage4_num_modules,
            self.stage4_num_blocks,
            self.stage4_num_channels,
            name='st4')

        # upsample
        shape = fluid.layers.shape(st4[0])[-2:]
        st4[1] = fluid.layers.resize_bilinear(st4[1], out_shape=shape)
        st4[2] = fluid.layers.resize_bilinear(st4[2], out_shape=shape)
        st4[3] = fluid.layers.resize_bilinear(st4[3], out_shape=shape)

        out = fluid.layers.concat(st4, axis=1)
        last_channels = sum(self.stage4_num_channels)

        out = self._conv_bn_layer(
            input=out,
            filter_size=1,
            num_filters=last_channels,
            stride=1,
            if_act=True,
            name='conv-2')
        out = fluid.layers.conv2d(
            input=out,
            num_filters=num_classes,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(initializer=MSRA(), name='conv-1_weights'),
            bias_attr=False)

        input_shape = fluid.layers.shape(input)[-2:]
        out = fluid.layers.resize_bilinear(out, input_shape)

        return out

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
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from .libs import sigmoid_to_softmax
from .seg_modules import softmax_with_loss
from .seg_modules import dice_loss
from .seg_modules import bce_loss


class ShuffleSeg(object):
    # def __init__(self):
    # self.params = train_parameters
    def __init__(self,
                 num_classes,
                 mode='train',
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
        self.mode = mode
        self.use_bce_loss = use_bce_loss
        self.use_dice_loss = use_dice_loss
        self.class_weight = class_weight
        self.ignore_index = ignore_index

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
        if self.use_dice_loss or self.use_bce_loss:
            self.num_classes = 1
        image = inputs['image']
        ## Encoder
        conv1 = self.conv_bn(image, 3, 36, 2, 1)
        print('encoder 1', conv1.shape)
        shortcut = self.conv_bn(
            input=conv1, filter_size=1, num_filters=18, stride=1, padding=0)
        print('shortcut 1', shortcut.shape)

        pool = fluid.layers.pool2d(
            input=conv1,
            pool_size=3,
            pool_type='max',
            pool_stride=2,
            pool_padding=1)
        print('encoder 2', pool.shape)

        # Block 1
        conv = self.sfnetv2module(pool, stride=2, num_filters=72)
        conv = self.sfnetv2module(conv, stride=1)
        conv = self.sfnetv2module(conv, stride=1)
        conv = self.sfnetv2module(conv, stride=1)
        print('encoder 3', conv.shape)

        # Block 2
        conv = self.sfnetv2module(conv, stride=2)
        conv = self.sfnetv2module(conv, stride=1)
        conv = self.sfnetv2module(conv, stride=1)
        conv = self.sfnetv2module(conv, stride=1)
        conv = self.sfnetv2module(conv, stride=1)
        conv = self.sfnetv2module(conv, stride=1)
        conv = self.sfnetv2module(conv, stride=1)
        conv = self.sfnetv2module(conv, stride=1)
        print('encoder 4', conv.shape)

        ### decoder
        conv = self.depthwise_separable(conv, 3, 64, 1)
        shortcut_shape = fluid.layers.shape(shortcut)[2:]
        conv_b = fluid.layers.resize_bilinear(conv, shortcut_shape)
        concat = fluid.layers.concat([shortcut, conv_b], axis=1)
        decode_conv = self.depthwise_separable(concat, 3, 64, 1)
        logit = self.output_layer(decode_conv, self.num_classes)

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

    def conv_bn(self,
                input,
                filter_size,
                num_filters,
                stride,
                padding,
                channels=None,
                num_groups=1,
                act='relu',
                use_cudnn=True):
        parameter_attr = ParamAttr(learning_rate=1, initializer=MSRA())
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=parameter_attr,
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)

    def depthwise_separable(self, input, filter_size, num_filters, stride):
        num_filters1 = int(input.shape[1])
        num_groups = num_filters1
        depthwise_conv = self.conv_bn(
            input=input,
            filter_size=filter_size,
            num_filters=int(num_filters1),
            stride=stride,
            padding=int(filter_size / 2),
            num_groups=num_groups,
            use_cudnn=False,
            act=None)

        pointwise_conv = self.conv_bn(
            input=depthwise_conv,
            filter_size=1,
            num_filters=num_filters,
            stride=1,
            padding=0)
        return pointwise_conv

    def sfnetv2module(self, input, stride, num_filters=None):
        if stride == 1:
            shortcut, branch = fluid.layers.split(
                input, num_or_sections=2, dim=1)
            if num_filters is None:
                in_channels = int(branch.shape[1])
            else:
                in_channels = int(num_filters / 2)
        else:
            branch = input
            if num_filters is None:
                in_channels = int(branch.shape[1])
            else:
                in_channels = int(num_filters / 2)
            shortcut = self.depthwise_separable(input, 3, in_channels, stride)
        branch_1x1 = self.conv_bn(
            input=branch,
            filter_size=1,
            num_filters=int(in_channels),
            stride=1,
            padding=0)
        branch_dw1x1 = self.depthwise_separable(branch_1x1, 3, in_channels,
                                                stride)
        output = fluid.layers.concat(input=[shortcut, branch_dw1x1], axis=1)

        # channel shuffle
        # b, c, h, w = output.shape
        shape = fluid.layers.shape(output)
        c = output.shape[1]
        b, h, w = shape[0], shape[2], shape[3]
        output = fluid.layers.reshape(x=output, shape=[b, 2, in_channels, h, w])
        output = fluid.layers.transpose(x=output, perm=[0, 2, 1, 3, 4])
        output = fluid.layers.reshape(x=output, shape=[b, c, h, w])
        return output

    def output_layer(self, input, out_dim):
        param_attr = fluid.param_attr.ParamAttr(
            learning_rate=1.,
            regularizer=fluid.regularizer.L2Decay(0.),
            initializer=fluid.initializer.Xavier())
        # deconv
        output = fluid.layers.conv2d_transpose(
            input=input,
            num_filters=out_dim,
            filter_size=2,
            padding=0,
            stride=2,
            bias_attr=True,
            param_attr=param_attr,
            act=None)
        return output

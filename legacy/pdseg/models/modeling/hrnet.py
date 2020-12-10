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

import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr

from utils.config import cfg


def conv_bn_layer(input,
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


def basic_block(input, num_filters, stride=1, downsample=False, name=None):
    residual = input
    conv = conv_bn_layer(
        input=input,
        filter_size=3,
        num_filters=num_filters,
        stride=stride,
        name=name + '_conv1')
    conv = conv_bn_layer(
        input=conv,
        filter_size=3,
        num_filters=num_filters,
        if_act=False,
        name=name + '_conv2')
    if downsample:
        residual = conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=num_filters,
            if_act=False,
            name=name + '_downsample')
    return fluid.layers.elementwise_add(x=residual, y=conv, act='relu')


def bottleneck_block(input, num_filters, stride=1, downsample=False, name=None):
    residual = input
    conv = conv_bn_layer(
        input=input,
        filter_size=1,
        num_filters=num_filters,
        name=name + '_conv1')
    conv = conv_bn_layer(
        input=conv,
        filter_size=3,
        num_filters=num_filters,
        stride=stride,
        name=name + '_conv2')
    conv = conv_bn_layer(
        input=conv,
        filter_size=1,
        num_filters=num_filters * 4,
        if_act=False,
        name=name + '_conv3')
    if downsample:
        residual = conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=num_filters * 4,
            if_act=False,
            name=name + '_downsample')
    return fluid.layers.elementwise_add(x=residual, y=conv, act='relu')


def fuse_layers(x, channels, multi_scale_output=True, name=None):
    out = []
    for i in range(len(channels) if multi_scale_output else 1):
        residual = x[i]
        shape = residual.shape
        width = shape[-1]
        height = shape[-2]
        for j in range(len(channels)):
            if j > i:
                y = conv_bn_layer(
                    x[j],
                    filter_size=1,
                    num_filters=channels[i],
                    if_act=False,
                    name=name + '_layer_' + str(i + 1) + '_' + str(j + 1))
                y = fluid.layers.resize_bilinear(
                    input=y, out_shape=[height, width])
                residual = fluid.layers.elementwise_add(
                    x=residual, y=y, act=None)
            elif j < i:
                y = x[j]
                for k in range(i - j):
                    if k == i - j - 1:
                        y = conv_bn_layer(
                            y,
                            filter_size=3,
                            num_filters=channels[i],
                            stride=2,
                            if_act=False,
                            name=name + '_layer_' + str(i + 1) + '_' +
                            str(j + 1) + '_' + str(k + 1))
                    else:
                        y = conv_bn_layer(
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


def branches(x, block_num, channels, name=None):
    out = []
    for i in range(len(channels)):
        residual = x[i]
        for j in range(block_num):
            residual = basic_block(
                residual,
                channels[i],
                name=name + '_branch_layer_' + str(i + 1) + '_' + str(j + 1))
        out.append(residual)
    return out


def high_resolution_module(x, channels, multi_scale_output=True, name=None):
    residual = branches(x, 4, channels, name=name)
    out = fuse_layers(
        residual, channels, multi_scale_output=multi_scale_output, name=name)
    return out


def transition_layer(x, in_channels, out_channels, name=None):
    num_in = len(in_channels)
    num_out = len(out_channels)
    out = []
    for i in range(num_out):
        if i < num_in:
            if in_channels[i] != out_channels[i]:
                residual = conv_bn_layer(
                    x[i],
                    filter_size=3,
                    num_filters=out_channels[i],
                    name=name + '_layer_' + str(i + 1))
                out.append(residual)
            else:
                out.append(x[i])
        else:
            residual = conv_bn_layer(
                x[-1],
                filter_size=3,
                num_filters=out_channels[i],
                stride=2,
                name=name + '_layer_' + str(i + 1))
            out.append(residual)
    return out


def stage(x, num_modules, channels, multi_scale_output=True, name=None):
    out = x
    for i in range(num_modules):
        if i == num_modules - 1 and multi_scale_output == False:
            out = high_resolution_module(
                out,
                channels,
                multi_scale_output=False,
                name=name + '_' + str(i + 1))
        else:
            out = high_resolution_module(
                out, channels, name=name + '_' + str(i + 1))

    return out


def layer1(input, name=None):
    conv = input
    for i in range(4):
        conv = bottleneck_block(
            conv,
            num_filters=64,
            downsample=True if i == 0 else False,
            name=name + '_' + str(i + 1))
    return conv


def high_resolution_net(input, num_classes):

    channels_2 = cfg.MODEL.HRNET.STAGE2.NUM_CHANNELS
    channels_3 = cfg.MODEL.HRNET.STAGE3.NUM_CHANNELS
    channels_4 = cfg.MODEL.HRNET.STAGE4.NUM_CHANNELS

    num_modules_2 = cfg.MODEL.HRNET.STAGE2.NUM_MODULES
    num_modules_3 = cfg.MODEL.HRNET.STAGE3.NUM_MODULES
    num_modules_4 = cfg.MODEL.HRNET.STAGE4.NUM_MODULES

    x = conv_bn_layer(
        input=input,
        filter_size=3,
        num_filters=64,
        stride=2,
        if_act=True,
        name='layer1_1')
    x = conv_bn_layer(
        input=x,
        filter_size=3,
        num_filters=64,
        stride=2,
        if_act=True,
        name='layer1_2')

    la1 = layer1(x, name='layer2')
    tr1 = transition_layer([la1], [256], channels_2, name='tr1')
    st2 = stage(tr1, num_modules_2, channels_2, name='st2')
    tr2 = transition_layer(st2, channels_2, channels_3, name='tr2')
    st3 = stage(tr2, num_modules_3, channels_3, name='st3')
    tr3 = transition_layer(st3, channels_3, channels_4, name='tr3')
    st4 = stage(tr3, num_modules_4, channels_4, name='st4')

    # upsample
    shape = st4[0].shape
    height, width = shape[-2], shape[-1]
    st4[1] = fluid.layers.resize_bilinear(st4[1], out_shape=[height, width])
    st4[2] = fluid.layers.resize_bilinear(st4[2], out_shape=[height, width])
    st4[3] = fluid.layers.resize_bilinear(st4[3], out_shape=[height, width])

    out = fluid.layers.concat(st4, axis=1)
    last_channels = sum(channels_4)

    out = conv_bn_layer(
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

    out = fluid.layers.resize_bilinear(out, input.shape[2:])

    return out


def hrnet(input, num_classes):
    logit = high_resolution_net(input, num_classes)
    return logit


if __name__ == '__main__':
    image_shape = [-1, 3, 769, 769]
    image = fluid.data(name='image', shape=image_shape, dtype='float32')
    logit = hrnet(image, 4)
    print("logit:", logit.shape)

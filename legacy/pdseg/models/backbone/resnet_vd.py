# coding: utf8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import math

import numpy as np
import paddle
import paddle.static as static
import paddle.nn.functional as F
from paddle import ParamAttr

__all__ = [
    "ResNet", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"
]

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


class ResNet():
    def __init__(self,
                 layers=50,
                 scale=1.0,
                 stem=None,
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0],
                 align_corners=True):
        self.params = train_parameters
        self.layers = layers
        self.scale = scale
        self.stem = stem
        self.lr_mult_list = lr_mult_list
        assert len(
            self.lr_mult_list
        ) == 5, "lr_mult_list length in ResNet must be 5 but got {}!!".format(
            len(self.lr_mult_list))
        self.curr_stage = 0
        self.align_corners = align_corners

    def net(self,
            input,
            class_dim=1000,
            end_points=None,
            decode_points=None,
            resize_points=None,
            dilation_dict=None):
        layers = self.layers
        supported_layers = [18, 34, 50, 101, 152]
        mult_grid = [1, 2, 4]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        decode_ends = dict()

        def check_points(count, points):
            if points is None:
                return False
            else:
                if isinstance(points, list):
                    return (True if count in points else False)
                else:
                    return (True if count == points else False)

        def get_dilated_rate(dilation_dict, idx):
            if dilation_dict is None or idx not in dilation_dict:
                return 1
            else:
                return dilation_dict[idx]

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        if self.stem == 'icnet' or self.stem == 'pspnet' or self.stem == 'deeplab':
            conv = self.conv_bn_layer(
                input=input,
                num_filters=int(32 * self.scale),
                filter_size=3,
                stride=2,
                act='relu',
                name="conv1_1")
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=int(32 * self.scale),
                filter_size=3,
                stride=1,
                act='relu',
                name="conv1_2")
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=int(64 * self.scale),
                filter_size=3,
                stride=1,
                act='relu',
                name="conv1_3")
        else:
            conv = self.conv_bn_layer(
                input=input,
                num_filters=int(64 * self.scale),
                filter_size=7,
                stride=2,
                act='relu',
                name="conv1")

        conv = F.max_pool2d(conv, kernel_size=3, stride=2, padding=1)

        layer_count = 1
        if check_points(layer_count, decode_points):
            decode_ends[layer_count] = conv

        if check_points(layer_count, end_points):
            return conv, decode_ends

        if layers >= 50:
            for block in range(len(depth)):
                self.curr_stage += 1
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    dilation_rate = get_dilated_rate(dilation_dict, block)
                    if block == 3:
                        dilation_rate = dilation_rate * mult_grid[i]

                    conv = self.bottleneck_block(
                        input=conv,
                        num_filters=int(num_filters[block] * self.scale),
                        stride=2
                        if i == 0 and block != 0 and dilation_rate == 1 else 1,
                        name=conv_name,
                        is_first=block == i == 0,
                        dilation=dilation_rate)
                    layer_count += 3

                    if check_points(layer_count, decode_points):
                        decode_ends[layer_count] = conv

                    if check_points(layer_count, end_points):
                        return conv, decode_ends

                    if check_points(layer_count, resize_points):
                        conv = self.interp(
                            conv,
                            np.ceil(
                                np.array(conv.shape[2:]).astype('int32') / 2))

            pool = F.adaptive_avg_pool2d(conv, output_size=(1, 1))
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = static.nn.fc(
                input=pool,
                size=class_dim,
                param_attr=ParamAttr(
                    initializer=paddle.nn.initializer.Uniform(-stdv, stdv)))
        else:
            for block in range(len(depth)):
                self.curr_stage += 1
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    conv = self.basic_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        is_first=block == i == 0,
                        name=conv_name)
                    layer_count += 2
                    if check_points(layer_count, decode_points):
                        decode_ends[layer_count] = conv

                    if check_points(layer_count, end_points):
                        return conv, decode_ends

            pool = F.adaptive_avg_pool2d(conv, output_size=(1, 1))
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = static.nn.fc(
                input=pool,
                size=class_dim,
                param_attr=ParamAttr(
                    initializer=paddle.nn.initializer.Uniform(-stdv, stdv)))
        return out

    def zero_padding(self, input, padding):
        return F.pad(input, [padding, padding, padding, padding])

    def interp(self, input, out_shape):
        out_shape = list(out_shape.astype("int32"))
        return F.interpolate(
            input, out_shape, mode='bilinear', align_corners=self.align_corners)

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      dilation=1,
                      groups=1,
                      act=None,
                      name=None):

        lr_mult = self.lr_mult_list[self.curr_stage]
        if self.stem == 'pspnet':
            bias_attr = ParamAttr(name=name + "_biases")
        else:
            bias_attr = False

        conv = static.nn.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2 if dilation == 1 else 0,
            dilation=dilation,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights", learning_rate=lr_mult),
            bias_attr=bias_attr,
            name=name + '.conv2d.output.1')

        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return static.nn.batch_norm(
            input=conv,
            act=act,
            name=bn_name + '.output.1',
            param_attr=ParamAttr(
                name=bn_name + '_scale', learning_rate=lr_mult),
            bias_attr=ParamAttr(bn_name + '_offset', learning_rate=lr_mult),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
        )

    def conv_bn_layer_new(self,
                          input,
                          num_filters,
                          filter_size,
                          stride=1,
                          groups=1,
                          act=None,
                          name=None):
        lr_mult = self.lr_mult_list[self.curr_stage]
        pool = F.avg_pool2d(
            input, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        conv = static.nn.conv2d(
            input=pool,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=1,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights", learning_rate=lr_mult),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return static.nn.batch_norm(
            input=conv,
            act=act,
            param_attr=ParamAttr(
                name=bn_name + '_scale', learning_rate=lr_mult),
            bias_attr=ParamAttr(bn_name + '_offset', learning_rate=lr_mult),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def shortcut(self, input, ch_out, stride, is_first, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            if is_first or stride == 1:
                return self.conv_bn_layer(input, ch_out, 1, stride, name=name)
            else:
                return self.conv_bn_layer_new(
                    input, ch_out, 1, stride, name=name)
        elif is_first:
            return self.conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(self,
                         input,
                         num_filters,
                         stride,
                         name,
                         is_first=False,
                         dilation=1):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            dilation=1,
            stride=1,
            act='relu',
            name=name + "_branch2a")
        if dilation > 1:
            conv0 = self.zero_padding(conv0, dilation)
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            dilation=dilation,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 4,
            dilation=1,
            filter_size=1,
            act=None,
            name=name + "_branch2c")

        short = self.shortcut(
            input,
            num_filters * 4,
            stride,
            is_first=is_first,
            name=name + "_branch1")

        return F.relu(short + conv2)

    def basic_block(self, input, num_filters, stride, is_first, name):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=3,
            act='relu',
            stride=stride,
            name=name + "_branch2a")
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            name=name + "_branch2b")
        short = self.shortcut(
            input, num_filters, stride, is_first, name=name + "_branch1")
        return F.relu(short + conv1)


def ResNet18():
    model = ResNet(layers=18)
    return model


def ResNet34():
    model = ResNet(layers=34)
    return model


def ResNet50():
    model = ResNet(layers=50)
    return model


def ResNet101():
    model = ResNet(layers=101)
    return model


def ResNet152():
    model = ResNet(layers=152)
    return model

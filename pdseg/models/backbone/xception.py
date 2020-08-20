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
import contextlib
import paddle
import math
import paddle.fluid as fluid
from models.libs.model_libs import scope, name_scope
from models.libs.model_libs import bn, bn_relu, relu
from models.libs.model_libs import conv
from models.libs.model_libs import separate_conv

__all__ = ['xception_65', 'xception_41', 'xception_71']


def check_data(data, number):
    if type(data) == int:
        return [data] * number
    assert len(data) == number
    return data


def check_stride(s, os):
    if s <= os:
        return True
    else:
        return False


def check_points(count, points):
    if points is None:
        return False
    else:
        if isinstance(points, list):
            return (True if count in points else False)
        else:
            return (True if count == points else False)


class Xception():
    def __init__(self, backbone="xception_65"):
        self.bottleneck_params = self.gen_bottleneck_params(backbone)
        self.backbone = backbone

    def gen_bottleneck_params(self, backbone='xception_65'):
        if backbone == 'xception_65':
            bottleneck_params = {
                "entry_flow": (3, [2, 2, 2], [128, 256, 728]),
                "middle_flow": (16, 1, 728),
                "exit_flow": (2, [2, 1], [[728, 1024, 1024], [1536, 1536,
                                                              2048]])
            }
        elif backbone == 'xception_41':
            bottleneck_params = {
                "entry_flow": (3, [2, 2, 2], [128, 256, 728]),
                "middle_flow": (8, 1, 728),
                "exit_flow": (2, [2, 1], [[728, 1024, 1024], [1536, 1536,
                                                              2048]])
            }
        elif backbone == 'xception_71':
            bottleneck_params = {
                "entry_flow": (5, [2, 1, 2, 1, 2], [128, 256, 256, 728, 728]),
                "middle_flow": (16, 1, 728),
                "exit_flow": (2, [2, 1], [[728, 1024, 1024], [1536, 1536,
                                                              2048]])
            }
        else:
            raise Exception(
                "xception backbont only support xception_41/xception_65/xception_71"
            )
        return bottleneck_params

    def net(self,
            input,
            output_stride=32,
            num_classes=1000,
            end_points=None,
            decode_points=None):
        self.stride = 2
        self.block_point = 0
        self.output_stride = output_stride
        self.decode_points = decode_points
        self.short_cuts = dict()
        with scope(self.backbone):
            # Entry flow
            data = self.entry_flow(input)
            if check_points(self.block_point, end_points):
                return data, self.short_cuts

            # Middle flow
            data = self.middle_flow(data)
            if check_points(self.block_point, end_points):
                return data, self.short_cuts

            # Exit flow
            data = self.exit_flow(data)
            if check_points(self.block_point, end_points):
                return data, self.short_cuts

            data = fluid.layers.reduce_mean(data, [2, 3], keep_dim=True)
            data = fluid.layers.dropout(data, 0.5)
            stdv = 1.0 / math.sqrt(data.shape[1] * 1.0)
            with scope("logit"):
                out = fluid.layers.fc(
                    input=data,
                    size=num_classes,
                    act='softmax',
                    param_attr=fluid.param_attr.ParamAttr(
                        name='weights',
                        initializer=fluid.initializer.Uniform(-stdv, stdv)),
                    bias_attr=fluid.param_attr.ParamAttr(name='bias'))

            return out

    def entry_flow(self, data):
        param_attr = fluid.ParamAttr(
            name=name_scope + 'weights',
            regularizer=None,
            initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.09))
        with scope("entry_flow"):
            with scope("conv1"):
                data = bn_relu(
                    conv(
                        data, 32, 3, stride=2, padding=1,
                        param_attr=param_attr))
            with scope("conv2"):
                data = bn_relu(
                    conv(
                        data, 64, 3, stride=1, padding=1,
                        param_attr=param_attr))

        # get entry flow params
        block_num = self.bottleneck_params["entry_flow"][0]
        strides = self.bottleneck_params["entry_flow"][1]
        chns = self.bottleneck_params["entry_flow"][2]
        strides = check_data(strides, block_num)
        chns = check_data(chns, block_num)

        # params to control your flow
        s = self.stride
        block_point = self.block_point
        output_stride = self.output_stride
        with scope("entry_flow"):
            for i in range(block_num):
                block_point = block_point + 1
                with scope("block" + str(i + 1)):
                    stride = strides[i] if check_stride(s * strides[i],
                                                        output_stride) else 1
                    data, short_cuts = self.xception_block(
                        data, chns[i], [1, 1, stride])
                    s = s * stride
                    if check_points(block_point, self.decode_points):
                        self.short_cuts[block_point] = short_cuts[1]

        self.stride = s
        self.block_point = block_point
        return data

    def middle_flow(self, data):
        block_num = self.bottleneck_params["middle_flow"][0]
        strides = self.bottleneck_params["middle_flow"][1]
        chns = self.bottleneck_params["middle_flow"][2]
        strides = check_data(strides, block_num)
        chns = check_data(chns, block_num)

        # params to control your flow
        s = self.stride
        block_point = self.block_point
        output_stride = self.output_stride
        with scope("middle_flow"):
            for i in range(block_num):
                block_point = block_point + 1
                with scope("block" + str(i + 1)):
                    stride = strides[i] if check_stride(s * strides[i],
                                                        output_stride) else 1
                    data, short_cuts = self.xception_block(
                        data, chns[i], [1, 1, strides[i]], skip_conv=False)
                    s = s * stride
                    if check_points(block_point, self.decode_points):
                        self.short_cuts[block_point] = short_cuts[1]

        self.stride = s
        self.block_point = block_point
        return data

    def exit_flow(self, data):
        block_num = self.bottleneck_params["exit_flow"][0]
        strides = self.bottleneck_params["exit_flow"][1]
        chns = self.bottleneck_params["exit_flow"][2]
        strides = check_data(strides, block_num)
        chns = check_data(chns, block_num)

        assert (block_num == 2)
        # params to control your flow
        s = self.stride
        block_point = self.block_point
        output_stride = self.output_stride
        with scope("exit_flow"):
            with scope('block1'):
                block_point += 1
                stride = strides[0] if check_stride(s * strides[0],
                                                    output_stride) else 1
                data, short_cuts = self.xception_block(data, chns[0],
                                                       [1, 1, stride])
                s = s * stride
                if check_points(block_point, self.decode_points):
                    self.short_cuts[block_point] = short_cuts[1]
            with scope('block2'):
                block_point += 1
                stride = strides[1] if check_stride(s * strides[1],
                                                    output_stride) else 1
                data, short_cuts = self.xception_block(
                    data,
                    chns[1], [1, 1, stride],
                    dilation=2,
                    has_skip=False,
                    activation_fn_in_separable_conv=True)
                s = s * stride
                if check_points(block_point, self.decode_points):
                    self.short_cuts[block_point] = short_cuts[1]

        self.stride = s
        self.block_point = block_point
        return data

    def xception_block(self,
                       input,
                       channels,
                       strides=1,
                       filters=3,
                       dilation=1,
                       skip_conv=True,
                       has_skip=True,
                       activation_fn_in_separable_conv=False):
        repeat_number = 3
        channels = check_data(channels, repeat_number)
        filters = check_data(filters, repeat_number)
        strides = check_data(strides, repeat_number)
        data = input
        results = []
        for i in range(repeat_number):
            with scope('separable_conv' + str(i + 1)):
                if not activation_fn_in_separable_conv:
                    data = relu(data)
                    data = separate_conv(
                        data,
                        channels[i],
                        strides[i],
                        filters[i],
                        dilation=dilation)
                else:
                    data = separate_conv(
                        data,
                        channels[i],
                        strides[i],
                        filters[i],
                        dilation=dilation,
                        act=relu)
                results.append(data)
        if not has_skip:
            return data, results
        if skip_conv:
            param_attr = fluid.ParamAttr(
                name=name_scope + 'weights',
                regularizer=None,
                initializer=fluid.initializer.TruncatedNormal(
                    loc=0.0, scale=0.09))
            with scope('shortcut'):
                skip = bn(
                    conv(
                        input,
                        channels[-1],
                        1,
                        strides[-1],
                        groups=1,
                        padding=0,
                        param_attr=param_attr))
        else:
            skip = input
        return data + skip, results


def xception_65():
    model = Xception("xception_65")
    return model


def xception_41():
    model = Xception("xception_41")
    return model


def xception_71():
    model = Xception("xception_71")
    return model


if __name__ == '__main__':
    image_shape = [-1, 3, 224, 224]
    image = fluid.data(name='image', shape=image_shape, dtype='float32')
    model = xception_65()
    logit = model.net(image)

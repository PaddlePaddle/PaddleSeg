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
from paddle.fluid import ParamAttr

__all__ = ["VGGNet"]


def check_points(count, points):
    if points is None:
        return False
    else:
        if isinstance(points, list):
            return (True if count in points else False)
        else:
            return (True if count == points else False)


class VGGNet():
    def __init__(self, layers=16):
        self.layers = layers

    def net(self, input, class_dim=1000, end_points=None, decode_points=None):
        short_cuts = dict()
        layers_count = 0
        layers = self.layers
        vgg_spec = {
            11: ([1, 1, 2, 2, 2]),
            13: ([2, 2, 2, 2, 2]),
            16: ([2, 2, 3, 3, 3]),
            19: ([2, 2, 4, 4, 4])
        }
        assert layers in vgg_spec.keys(), \
            "supported layers are {} but input layer is {}".format(vgg_spec.keys(), layers)

        nums = vgg_spec[layers]
        channels = [64, 128, 256, 512, 512]
        conv = input
        for i in range(len(nums)):
            conv = self.conv_block(
                conv, channels[i], nums[i], name="conv" + str(i + 1) + "_")
            layers_count += nums[i]
            if check_points(layers_count, decode_points):
                short_cuts[layers_count] = conv
            if check_points(layers_count, end_points):
                return conv, short_cuts

        return conv

    def conv_block(self, input, num_filter, groups, name=None):
        conv = input
        for i in range(groups):
            conv = fluid.layers.conv2d(
                input=conv,
                num_filters=num_filter,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(
                    name=name + str(i + 1) + "_weights"),
                bias_attr=False)
        return fluid.layers.pool2d(
            input=conv, pool_size=2, pool_type='max', pool_stride=2)

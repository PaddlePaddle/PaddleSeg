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

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddleslim.nas.search_space.search_space_base import SearchSpaceBase
from paddleslim.nas.search_space.base_layer import conv_bn_layer
from paddleslim.nas.search_space.search_space_registry import SEARCHSPACE
from paddleslim.nas.search_space.utils import check_points

__all__ = ["MobileNetV2SpaceSeg"]


@SEARCHSPACE.register
class MobileNetV2SpaceSeg(SearchSpaceBase):
    def __init__(self, input_size, output_size, block_num, block_mask=None):
        super(MobileNetV2SpaceSeg, self).__init__(input_size, output_size,
                                                  block_num, block_mask)
        # self.head_num means the first convolution channel
        self.head_num = np.array([3, 4, 8, 12, 16, 24, 32])  #7
        # self.filter_num1 ~ self.filter_num6 means following convlution channel
        self.filter_num1 = np.array([3, 4, 8, 12, 16, 24, 32, 48])  #8
        self.filter_num2 = np.array([8, 12, 16, 24, 32, 48, 64, 80])  #8
        self.filter_num3 = np.array([16, 24, 32, 48, 64, 80, 96, 128])  #8
        self.filter_num4 = np.array(
            [24, 32, 48, 64, 80, 96, 128, 144, 160, 192])  #10
        self.filter_num5 = np.array(
            [32, 48, 64, 80, 96, 128, 144, 160, 192, 224])  #10
        self.filter_num6 = np.array(
            [64, 80, 96, 128, 144, 160, 192, 224, 256, 320, 384, 512])  #12
        # self.k_size means kernel size
        self.k_size = np.array([3, 5])  #2
        # self.multiply means expansion_factor of each _inverted_residual_unit
        self.multiply = np.array([1, 2, 3, 4, 6])  #5
        # self.repeat means repeat_num _inverted_residual_unit in each _invresi_blocks
        self.repeat = np.array([1, 2, 3, 4, 5, 6])  #6

    def init_tokens(self):
        """
        The initial token.
        The first one is the index of the first layers' channel in self.head_num,
        each line in the following represent the index of the [expansion_factor, filter_num, repeat_num, kernel_size]
        """
        # original MobileNetV2
        # yapf: disable
        init_token_base =  [4,          # 1, 16, 1
                4, 5, 1, 0, # 6, 24, 2
                4, 4, 2, 0, # 6, 32, 3
                4, 4, 3, 0, # 6, 64, 4
                4, 5, 2, 0, # 6, 96, 3
                4, 7, 2, 0, # 6, 160, 3
                4, 9, 0, 0] # 6, 320, 1
        # yapf: enable

        return init_token_base

    def range_table(self):
        """
        Get range table of current search space, constrains the range of tokens.
        """
        # head_num + 6 * [multiple(expansion_factor), filter_num, repeat, kernel_size]
        # yapf: disable
        range_table_base =  [len(self.head_num),
                len(self.multiply), len(self.filter_num1), len(self.repeat), len(self.k_size),
                len(self.multiply), len(self.filter_num2), len(self.repeat), len(self.k_size),
                len(self.multiply), len(self.filter_num3), len(self.repeat), len(self.k_size),
                len(self.multiply), len(self.filter_num4), len(self.repeat), len(self.k_size),
                len(self.multiply), len(self.filter_num5), len(self.repeat), len(self.k_size),
                len(self.multiply), len(self.filter_num6), len(self.repeat), len(self.k_size)]
        # yapf: enable
        return range_table_base

    def token2arch(self, tokens=None):
        """
        return net_arch function
        """

        if tokens is None:
            tokens = self.init_tokens()

        self.bottleneck_params_list = []
        self.bottleneck_params_list.append((1, self.head_num[tokens[0]], 1, 1,
                                            3))
        self.bottleneck_params_list.append(
            (self.multiply[tokens[1]], self.filter_num1[tokens[2]],
             self.repeat[tokens[3]], 2, self.k_size[tokens[4]]))
        self.bottleneck_params_list.append(
            (self.multiply[tokens[5]], self.filter_num2[tokens[6]],
             self.repeat[tokens[7]], 2, self.k_size[tokens[8]]))
        self.bottleneck_params_list.append(
            (self.multiply[tokens[9]], self.filter_num3[tokens[10]],
             self.repeat[tokens[11]], 2, self.k_size[tokens[12]]))
        self.bottleneck_params_list.append(
            (self.multiply[tokens[13]], self.filter_num4[tokens[14]],
             self.repeat[tokens[15]], 1, self.k_size[tokens[16]]))
        self.bottleneck_params_list.append(
            (self.multiply[tokens[17]], self.filter_num5[tokens[18]],
             self.repeat[tokens[19]], 2, self.k_size[tokens[20]]))
        self.bottleneck_params_list.append(
            (self.multiply[tokens[21]], self.filter_num6[tokens[22]],
             self.repeat[tokens[23]], 1, self.k_size[tokens[24]]))

        def _modify_bottle_params(output_stride=None):
            if output_stride is not None and output_stride % 2 != 0:
                raise Exception("output stride must to be even number")
            if output_stride is None:
                return
            else:
                stride = 2
                for i, layer_setting in enumerate(self.bottleneck_params_list):
                    t, c, n, s, ks = layer_setting
                    stride = stride * s
                    if stride > output_stride:
                        s = 1
                    self.bottleneck_params_list[i] = (t, c, n, s, ks)

        def net_arch(input,
                     scale=1.0,
                     return_block=None,
                     end_points=None,
                     output_stride=None):
            self.scale = scale
            _modify_bottle_params(output_stride)

            decode_ends = dict()

            def check_points(count, points):
                if points is None:
                    return False
                else:
                    if isinstance(points, list):
                        return (True if count in points else False)
                    else:
                        return (True if count == points else False)

            #conv1
            # all padding is 'SAME' in the conv2d, can compute the actual padding automatic.
            input = conv_bn_layer(
                input,
                num_filters=int(32 * self.scale),
                filter_size=3,
                stride=2,
                padding='SAME',
                act='relu6',
                name='mobilenetv2_conv1')
            layer_count = 1

            depthwise_output = None
            # bottleneck sequences
            in_c = int(32 * self.scale)
            for i, layer_setting in enumerate(self.bottleneck_params_list):
                t, c, n, s, k = layer_setting
                layer_count += 1
                ### return_block and end_points means block num
                if check_points((layer_count - 1), return_block):
                    decode_ends[layer_count - 1] = depthwise_output

                if check_points((layer_count - 1), end_points):
                    return input, decode_ends
                input, depthwise_output = self._invresi_blocks(
                    input=input,
                    in_c=in_c,
                    t=t,
                    c=int(c * self.scale),
                    n=n,
                    s=s,
                    k=int(k),
                    name='mobilenetv2_conv' + str(i))
                in_c = int(c * self.scale)

            ### return_block and end_points means block num
            if check_points(layer_count, return_block):
                decode_ends[layer_count] = depthwise_output

            if check_points(layer_count, end_points):
                return input, decode_ends
            # last conv
            input = conv_bn_layer(
                input=input,
                num_filters=int(1280 * self.scale)
                if self.scale > 1.0 else 1280,
                filter_size=1,
                stride=1,
                padding='SAME',
                act='relu6',
                name='mobilenetv2_conv' + str(i + 1))

            input = fluid.layers.pool2d(
                input=input,
                pool_type='avg',
                global_pooling=True,
                name='mobilenetv2_last_pool')

            return input

        return net_arch

    def _shortcut(self, input, data_residual):
        """Build shortcut layer.
        Args:
            input(Variable): input.
            data_residual(Variable): residual layer.
        Returns:
            Variable, layer output.
        """
        return fluid.layers.elementwise_add(input, data_residual)

    def _inverted_residual_unit(self,
                                input,
                                num_in_filter,
                                num_filters,
                                ifshortcut,
                                stride,
                                filter_size,
                                expansion_factor,
                                reduction_ratio=4,
                                name=None):
        """Build inverted residual unit.
        Args:
            input(Variable), input.
            num_in_filter(int), number of in filters.
            num_filters(int), number of filters.
            ifshortcut(bool), whether using shortcut.
            stride(int), stride.
            filter_size(int), filter size.
            padding(str|int|list), padding.
            expansion_factor(float), expansion factor.
            name(str), name.
        Returns:
            Variable, layers output.
        """
        num_expfilter = int(round(num_in_filter * expansion_factor))
        channel_expand = conv_bn_layer(
            input=input,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding='SAME',
            num_groups=1,
            act='relu6',
            name=name + '_expand')

        bottleneck_conv = conv_bn_layer(
            input=channel_expand,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding='SAME',
            num_groups=num_expfilter,
            act='relu6',
            name=name + '_dwise',
            use_cudnn=False)

        depthwise_output = bottleneck_conv

        linear_out = conv_bn_layer(
            input=bottleneck_conv,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding='SAME',
            num_groups=1,
            act=None,
            name=name + '_linear')
        out = linear_out
        if ifshortcut:
            out = self._shortcut(input=input, data_residual=out)
        return out, depthwise_output

    def _invresi_blocks(self, input, in_c, t, c, n, s, k, name=None):
        """Build inverted residual blocks.
        Args:
            input: Variable, input.
            in_c: int, number of in filters.
            t: float, expansion factor.
            c: int, number of filters.
            n: int, number of layers.
            s: int, stride.
            k: int, filter size.
            name: str, name.
        Returns:
            Variable, layers output.
        """
        first_block, depthwise_output = self._inverted_residual_unit(
            input=input,
            num_in_filter=in_c,
            num_filters=c,
            ifshortcut=False,
            stride=s,
            filter_size=k,
            expansion_factor=t,
            name=name + '_1')

        last_residual_block = first_block
        last_c = c

        for i in range(1, n):
            last_residual_block, depthwise_output = self._inverted_residual_unit(
                input=last_residual_block,
                num_in_filter=last_c,
                num_filters=c,
                ifshortcut=True,
                stride=1,
                filter_size=k,
                expansion_factor=t,
                name=name + '_' + str(i + 1))
        return last_residual_block, depthwise_output

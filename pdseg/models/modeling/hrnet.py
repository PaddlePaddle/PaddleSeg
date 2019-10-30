# coding: utf8
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import sys
sys.path.append('../../')

import paddle.fluid as fluid
from models.libs import model_libs
from paddle.fluid.param_attr import ParamAttr
from models.libs.model_libs import scope
from models.libs.model_libs import avg_pool, conv, bn, bn_relu, relu
from models.backbone.resnet import ResNet
from utils.config import cfg

import logging

# resnet = ResNet(stem='pspnet', layers=101)
resnet = ResNet()
bottleneck_block = resnet.bottleneck_block
basic_block = resnet.basic_block

def make_one_branch(input, branch_index, num_blocks, num_channels, stride=1):
    print(num_blocks)
    for i in range(num_blocks[branch_index]):
        with scope('basic_' + str(i)):
            input = basic_block(
                input,
                num_channels[branch_index],
                stride=1,
                is_first=False,
                name=model_libs.name_scope)
    return input

def make_branches(input, num_branches, num_blocks, num_channels):
    '''
    各分支各自进行卷积操作
    Args:
        input: 输入，为一个列表
        num_branches: 分支数
        num_blocks:
        num_inchannels:
        num_channels:
    Returns:
        返回一个列表
    '''

    branches = []
    for i in range(num_branches):
        with scope('branch_' + str(i)):
            branches.append(make_one_branch(input[i], i, num_blocks, num_channels))
    return branches

def make_fuse_layer(input, num_branches, num_channels):
    if num_branches == 1:
        return input

    fuse_layers = []
    for i in range(num_branches):
        fuse_layer = []
        shape = input[i].shape
        width = shape[-1]
        height = shape[-2]
        for j in range(num_branches):
            if j > i:
                with scope('up1x1'+str(j)+str(i)):
                    fuse_j2i = (bn(conv(
                                input[j],
                                num_channels[i],
                                1,
                                stride=1,
                                padding=0)))
                    fuse_j2i = fluid.layers.resize_bilinear(
                        fuse_j2i, out_shape=[height, width])
                    fuse_layer.append(fuse_j2i)

            elif j == i:
                fuse_layer.append(input[i])

            else:
                fuse_j2i = input[j]
                for k in range(i-j):
                    if k == i - j - 1:
                        with scope('conv' + str(j) + str(i) + str(k)):
                            fuse_j2i = (bn(conv(
                                fuse_j2i,
                                num_channels[i],
                                3,
                                stride=2,
                                padding=1)))
                    else:
                        with scope('conv' + str(j) + str(i) + str(k)):
                            fuse_j2i = (bn_relu(conv(
                                fuse_j2i,
                                num_channels[j],
                                3,
                                stride=2,
                                padding=1)))
                fuse_layer.append(fuse_j2i)

        fuse_layers.append(fluid.layers.sum(fuse_layer))
    return fuse_layers

def highResolutionModule(input, num_branches, num_blocks, num_channels):
    '''
    实现HRNet的一个模块，包括各个分支自身的卷积，以及各分支之间的融合

    Args:
        input: 输入，为一个列表
        num_branches: 分支数
        num_blocks: 进行多少次basic_blocks/bottleneck_block
        num_channels: 每一个分支的通道数
    Returns:
        返回一个列表
    '''

    print('input')
    for i in input:
        print('check:')
        print(i.shape)
    if num_branches == 1:
        return make_branches(input, num_branches, num_blocks, num_channels)
    input = make_branches(input, num_branches, num_blocks, num_channels)
    print('output')
    for i in input:
        print('check:')
        print(i.shape)
    input = make_fuse_layer(input, num_branches, num_channels)

    print('output')
    for i in input:
        print('check:')
        print(i.shape)
    return input





def make_transition_layer(input, num_channels_pre_layer, num_channels_cur_layer):
    '''
    实现两个阶段的衔接，实现分支数的增加

    Args
        input: 上一阶段的输出，为一个列表，表示每一个分支的输出
        num_channels_pre_layer:
        num_channels_cur_layer:

    Returns:
        返回一个列表，表示下一个阶段的输入
    '''
    num_branches_pre = len(num_channels_pre_layer)
    num_branches_cur = len(num_channels_cur_layer)

    y_list = []
    for i in range(num_branches_cur):
        with scope('conv' + str(i)):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    y_list.append(bn_relu(
                        conv(
                            input[i],
                            num_channels_cur_layer[i],
                            3,
                            stride=1,
                            padding=1)))
                else:
                    y_list.append(input[i])
            else:
                y_list.append(conv(
                    input[-1],
                    num_channels_cur_layer[i],
                    3,
                    stride=2,
                    padding=1))
    return y_list

def make_stage(input, layer_config):
    num_modules = layer_config.NUM_MODULES
    num_branches = layer_config.NUM_BRANCHES
    num_blocks = layer_config.NUM_BLOCKS
    num_channels = layer_config.NUM_CHANNELS
    print(num_modules, num_branches, num_blocks, num_channels)
    for i in range(num_modules):
        with scope('module_' + str(i)):
            input = highResolutionModule(input,
                                       num_branches,
                                       num_blocks,
                                       num_channels)
    # num_inchannels = [input[i].shape[1] for i in range(num_branches)]
    return input, num_channels





def highResolutionNet(input):
    param_attr = ParamAttr(
        name=model_libs.name_scope + 'weights',
        regularizer=None,
        initializer=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.06))
    with scope('first_conv1'):
        logit = bn_relu(
            conv(
                input,
                64,
                3,
                stride=2,
                padding=1,
                param_attr=param_attr))
    with scope('first_conv2'):
        logit = bn_relu(
            conv(
                logit,
                64,
                3,
                stride=2,
                padding=1,
                param_attr=param_attr))

    # 第一阶段
    with scope('layer1'):
        num_filters = cfg.MODEL.HRNET.STAGE1.NUM_CHANNELS[0]
        blocks = cfg.MODEL.HRNET.STAGE1.NUM_BLOCKS[0]
        for i in range(blocks):
            with scope('bottlenet_' + str(i)):
                logit = bottleneck_block(
                    logit,
                    num_filters,
                    stride=1,
                    name=model_libs.name_scope)
    stage1_out_channel = num_filters

    # 第二阶段
    with scope('stage2'):
        num_filters = cfg.MODEL.HRNET.STAGE2.NUM_CHANNELS
        with scope('transition'):
            stage2_input_list = make_transition_layer([logit],
                                                     [stage1_out_channel],
                                                     num_filters)
        stage2_config = cfg.MODEL.HRNET.STAGE2
        stage2_output_list, pre_stage_channels = make_stage(
            stage2_input_list,
            stage2_config)

        print('stage2')
        for i in stage2_output_list:
            print(i.shape)

    # 第三阶段
    with scope('stage3'):
        num_filters = cfg.MODEL.HRNET.STAGE3.NUM_CHANNELS
        with scope('transition'):
            stage3_input_list = make_transition_layer(stage2_output_list,
                                                      pre_stage_channels,
                                                      num_filters)
        stage3_config = cfg.MODEL.HRNET.STAGE3
        stage3_output_list, pre_stage_channels = make_stage(
            stage3_input_list,
            stage3_config)

        print('stage3')
        for i in stage3_output_list:
            print(i.shape)

    # 第四阶段
    with scope('stage4'):
        num_filters = cfg.MODEL.HRNET.STAGE4.NUM_CHANNELS
        with scope('transition'):
            stage4_input_list = make_transition_layer(stage3_output_list,
                                                      pre_stage_channels,
                                                      num_filters)
        stage4_config = cfg.MODEL.HRNET.STAGE4
        stage4_output_list, pre_stage_channels = make_stage(
            stage4_input_list,
            stage4_config)

        print('stage4')
        for i in stage4_output_list:
            print(i.shape)

    # upsampling
    shape = stage4_output_list[0].shape
    height, width = shape[-2], shape[-1]
    stage4_output_list[1] = fluid.layers.resize_bilinear(
        stage4_output_list[1], out_shape=[height, width])
    stage4_output_list[2] = fluid.layers.resize_bilinear(
        stage4_output_list[2], out_shape=[height, width])
    stage4_output_list[3] = fluid.layers.resize_bilinear(
        stage4_output_list[3], out_shape=[height, width])

    logit = fluid.layers.concat(stage4_output_list, axis=1)
    last_channels = sum(pre_stage_channels)

    with scope('last_conv2'):
        logit = bn_relu(conv(
            logit,
            last_channels,
            filter_size=1,
            stride=1,
            padding=0
        ))

    with scope('last_conv1'):
        logit = conv(
            logit,
            cfg.DATASET.NUM_CLASSES,
            filter_size=1,
            stride=1,
            padding=0
        )

    return logit


def hrnet(input, num_classes):
    logit = highResolutionNet(input)
    return logit

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(message)s"
    formatter = logging.Formatter(BASIC_FORMAT)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    image_shape = [3, 512, 1024]
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    logit = hrnet(image, 4)
    logger.info("logit:", logit.shape)
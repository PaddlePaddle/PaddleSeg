# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# Original head information
"""
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
"""

import os
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import param_init
from paddleseg.utils import logger, utils

from paddlepanseg.cvlibs import manager

BN_MOMENTUM = 0.99

__all__ = ['HRNet']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding=1,
        bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(
            inplanes, planes, kernel_size=(1, 1), bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2D(
            planes,
            planes,
            kernel_size=(3, 3),
            stride=(stride, stride),
            padding=1,
            bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2D(
            planes,
            planes * self.expansion,
            kernel_size=(1, 1),
            bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Layer):
    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 num_in_channels,
                 num_channels,
                 multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_in_channels,
                             num_channels)

        self.num_in_channels = num_in_channels
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()

    def _check_branches(self, num_branches, blocks, num_blocks, num_in_channels,
                        num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'num_branches ({}) != num_blocks ({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'num_branches ({}) <> num_channels ({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_in_channels):
            error_msg = 'num_branches ({}) <> num_in_channels ({})'.format(
                num_branches, len(num_in_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_in_channels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.num_in_channels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False),
                nn.BatchNorm2D(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM), )

        layers = []
        layers.append(
            block(self.num_in_channels[branch_index], num_channels[
                branch_index], stride, downsample))
        self.num_in_channels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_in_channels[branch_index], num_channels[
                    branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.LayerList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_in_channels = self.num_in_channels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2D(
                                num_in_channels[j],
                                num_in_channels[i],
                                1,
                                1,
                                0,
                                bias_attr=False),
                            nn.BatchNorm2D(
                                num_in_channels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_in_channels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2D(
                                        num_in_channels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias_attr=False),
                                    nn.BatchNorm2D(
                                        num_outchannels_conv3x3,
                                        momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_in_channels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2D(
                                        num_in_channels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias_attr=False),
                                    nn.BatchNorm2D(
                                        num_outchannels_conv3x3,
                                        momentum=BN_MOMENTUM),
                                    nn.ReLU()))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.LayerList(fuse_layer))

        return nn.LayerList(fuse_layers)

    def get_num_in_channels(self):
        return self.num_in_channels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {'basic': BasicBlock, 'bottleneck': Bottleneck}


class HighResolutionNet(nn.Layer):
    def __init__(self,
                 in_channels,
                 base_channels,
                 block_type,
                 num_blocks,
                 num_branches,
                 num_modules,
                 pretrained=None):
        super(HighResolutionNet, self).__init__()

        # Stem net
        self.conv1 = nn.Conv2D(
            in_channels,
            64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2D(
            64,
            64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias_attr=False)
        self.bn2 = nn.BatchNorm2D(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

        num_channels = base_channels[0]
        block = blocks_dict[block_type[0]]
        n_blocks = num_blocks[0]
        self.layer1 = self._make_layer(block, 64, num_channels, n_blocks)
        stage1_out_channel = block.expansion * num_channels

        base_channel = base_channels[1]
        block = blocks_dict[block_type[1]]
        num_channels = [base_channel * 2**i for i in range(2)]
        num_in_channels = [num_channels[i] * block.expansion for i in range(2)]
        self.stage2_num_branches = num_branches[1]
        self.transition1 = self._make_transition_layer([stage1_out_channel],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            block=block,
            num_modules=num_modules[1],
            num_branches=self.stage2_num_branches,
            num_blocks=[num_blocks[1]] * 2,
            num_in_channels=num_in_channels,
            num_channels=num_channels)

        base_channel = base_channels[2]
        block = blocks_dict[block_type[2]]
        num_channels = [base_channel * 2**i for i in range(3)]
        num_in_channels = [num_channels[i] * block.expansion for i in range(3)]
        self.stage3_num_branches = num_branches[2]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            block=block,
            num_modules=num_modules[2],
            num_branches=self.stage3_num_branches,
            num_blocks=[num_blocks[2]] * 3,
            num_in_channels=num_in_channels,
            num_channels=num_channels)

        base_channel = base_channels[3]
        block = blocks_dict[block_type[3]]
        num_channels = [base_channel * 2**i for i in range(4)]
        num_in_channels = [num_channels[i] * block.expansion for i in range(4)]
        self.stage4_num_branches = num_branches[3]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            block=block,
            num_modules=num_modules[3],
            num_branches=self.stage4_num_branches,
            num_blocks=[num_blocks[3]] * 4,
            num_in_channels=num_in_channels,
            num_channels=num_channels,
            multi_scale_output=True)

        self.feat_channels = num_channels

        self.pretrained = pretrained
        self.init_weight()

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2D(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i], (3, 3), (1, 1),
                                1,
                                bias_attr=False),
                            nn.BatchNorm2D(
                                num_channels_cur_layer[i],
                                momentum=BN_MOMENTUM),
                            nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2D(
                                inchannels,
                                outchannels, (3, 3), (2, 2),
                                1,
                                bias_attr=False),
                            nn.BatchNorm2D(
                                outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU()))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.LayerList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=(1, 1),
                    stride=(stride, stride),
                    bias_attr=False),
                nn.BatchNorm2D(
                    planes * block.expansion, momentum=BN_MOMENTUM), )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self,
                    block,
                    num_modules,
                    num_branches,
                    num_blocks,
                    num_in_channels,
                    num_channels,
                    multi_scale_output=True):

        modules = []
        for i in range(num_modules):
            # `multi_scale_output` is only used in last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches, block, num_blocks,
                                     num_in_channels, num_channels,
                                     reset_multi_scale_output))
            num_in_channels = modules[-1].get_num_in_channels()

        return nn.Sequential(*modules), num_in_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_num_branches):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        return x

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                n = layer._kernel_size[0] * layer._kernel_size[
                    1] * layer._out_channels
                param_init.normal_init(layer.weight, std=math.sqrt(2. / n))
                if layer.bias is not None:
                    param_init.constant_init(layer.bias, value=0)
            elif isinstance(layer, nn.BatchNorm2D):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
            elif isinstance(layer, nn.Linear):
                param_init.normal_init(layer.weight, std=0.001)
                if layer.bias is not None:
                    param_init.constant_init(layer.bias, value=0)
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


@manager.BACKBONES.add_component
def HRNet(**kwargs):
    model = HighResolutionNet(**kwargs)
    return model

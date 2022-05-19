# Most of the code below is from the following repo:
#  https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR
#
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .PSA import PSA_p, PSA_s

import os
import numpy as np
from .recompute import recompute
import random
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .mynn import Norm2d
#from runx.logx import logx
from .config import cfg
from .para_init import kaiming_normal_,kaiming_uniform_,constant_

BN_MOMENTUM = 0.1
align_corners = cfg.MODEL.ALIGN_CORNERS
#relu_inplace = True

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = Norm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.deattn = PSA_s(planes, planes)
        #self.deattn = PSA_p(planes, planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = Norm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.deattn(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = Norm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = Norm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2D(planes, planes * self.expansion, kernel_size=1,
                               bias_attr=False)
        self.bn3 = Norm2d(planes * self.expansion, momentum=BN_MOMENTUM)
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

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Layer):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            #logx.msg(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            #logx.msg(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            #logx.msg(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != (num_channels[branch_index] *
                                                 block.expansion):
            downsample = nn.Sequential(
                nn.Conv2D(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                Norm2d(num_channels[branch_index] * block.expansion,
                       momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

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
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2D(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias_attr=False),
                        Norm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2D(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias_attr=False),
                                Norm2d(num_outchannels_conv3x3,
                                       momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2D(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias_attr=False),
                                Norm2d(num_outchannels_conv3x3,
                                       momentum=BN_MOMENTUM),
                                nn.ReLU()))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
   
            fuse_layers.append(nn.LayerList(fuse_layer))
        return nn.LayerList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            #return [self.branches[0](x[0])]
            return [recompute(self.branches[0],x[0])]

        for i in range(self.num_branches):
            #x[i] = self.branches[i](x[i])
            x[i] = recompute(self.branches[i],x[i])


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
                        mode='bilinear', align_corners=align_corners)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Layer):

    def __init__(self, **kwargs):
        extra = cfg.MODEL.OCR_EXTRA
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2D(3, 64, kernel_size=3, stride=2, padding=1,
                               bias_attr=False)
        self.bn1 = Norm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2D(64, 64, kernel_size=3, stride=2, padding=1,
                               bias_attr=False)
        self.bn2 = Norm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion
                        for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion
                        for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion
                        for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        self.high_level_ch = np.int(np.sum(pre_stage_channels))

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2D(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias_attr=False),
                        Norm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2D(
                            inchannels, outchannels, 3, 2, 1, bias_attr=False),
                        Norm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU()))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.LayerList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                Norm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        #x = recompute(self.stage4,x_list)

        # Upsampling
        #x0_h, x0_w = x[0].size(2), x[0].size(3)
        x0_h, x0_w = x[0].shape[2], x[0].shape[3]
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=align_corners)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=align_corners)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=align_corners)

        feats = paddle.concat([x[0], x1, x2, x3], 1)

        return None, None, feats

    def init_weights(self, pretrained='pretrained/hrnetv2_w48_imagenet_pretrained.pdparams'):
        #logx.msg('=> init weights from normal distribution')
        for name, m in self.named_children():
            if any(part in name for part in {'cls', 'aux', 'ocr'}):
                # print('skipped', name)
                continue
            if isinstance(m, nn.Conv2D):
                #修改初始化
                #nn.init.normal_(m.weight, std=0.001)
                initializer = paddle.nn.initializer.Normal(std=0.001)
                initializer(m.weight)
            elif isinstance(m, cfg.MODEL.BNFUNC):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
        if os.path.isfile(pretrained):

            #修改后的加载模型
            pretrained_dict = paddle.load(pretrained)
            model_dict = self.state_dict()
            pretrained_dict = {k.replace('last_layer','aux_head').replace('model.', ''): v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_dict(model_dict)
            
                      
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))


def get_seg_model():
    model = HighResolutionNet()
    model.init_weights()

    return model

# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import math
import warnings
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.utils import utils
from paddleseg.cvlibs import manager
from paddleseg.cvlibs import param_init
from paddleseg.models import layers


class PolarizedSelfAttentionModule_p(nn.Layer):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super().__init__()
        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.conv_q_right = nn.Conv2D(
            self.inplanes,
            1,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias_attr=False)
        self.conv_v_right = nn.Conv2D(
            self.inplanes,
            self.inter_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias_attr=False)
        self.conv_up = nn.Conv2D(
            self.inter_planes,
            self.planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False)
        self.softmax_right = nn.Softmax(axis=2)
        self.sigmoid = nn.Sigmoid()
        self.conv_q_left = nn.Conv2D(
            self.inplanes,
            self.inter_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias_attr=False)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv_v_left = nn.Conv2D(
            self.inplanes,
            self.inter_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias_attr=False)
        self.softmax_left = nn.Softmax(axis=2)

        self.reset_parameters()

    def reset_parameters(self):
        self.kaiming_init(self.conv_q_right, mode='fan_in')
        self.kaiming_init(self.conv_v_right, mode='fan_in')
        self.kaiming_init(self.conv_q_left, mode='fan_in')
        self.kaiming_init(self.conv_v_left, mode='fan_in')
        self.kaiming_init(self.conv_up, mode='fan_in')
        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True
        self.conv_up.inited = True

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)
        batch, _, height, width = paddle.shape(input_x)
        input_x = input_x.reshape((batch, self.inter_planes, height * width))
        context_mask = self.conv_q_right(x)
        context_mask = context_mask.reshape((batch, 1, height * width))
        context_mask = self.softmax_right(context_mask)
        context = paddle.matmul(input_x, context_mask.transpose((0, 2, 1)))
        context = context.unsqueeze(-1)
        context = self.conv_up(context)
        mask_ch = self.sigmoid(context)
        out = x * mask_ch
        return out

    def channel_pool(self, x):
        g_x = self.conv_q_left(x)
        batch, channel, height, width = paddle.shape(g_x)
        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = paddle.shape(avg_x)
        avg_x = avg_x.reshape((batch, channel, avg_x_h * avg_x_w))
        avg_x = paddle.reshape(avg_x, [batch, avg_x_h * avg_x_w, channel])
        theta_x = self.conv_v_left(x).reshape(
            (batch, self.inter_planes, height * width))
        context = paddle.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.reshape((batch, 1, height, width))
        mask_sp = self.sigmoid(context)
        out = x * mask_sp
        return out

    def forward(self, x):
        context_channel = self.spatial_pool(x)
        context_spatial = self.channel_pool(x)
        out = context_spatial + context_channel
        return out

    def calculate_gain(self, nonlinearity, param=None):
        linear_fns = [
            'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
            'conv_transpose2d', 'conv_transpose3d'
        ]
        if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
            return 1
        elif nonlinearity == 'tanh':
            return 5.0 / 3
        elif nonlinearity == 'relu':
            return math.sqrt(2.0)
        elif nonlinearity == 'leaky_relu':
            if param is None:
                negative_slope = 0.01
            elif not isinstance(param, bool) and isinstance(
                    param, int) or isinstance(param, float):
                # True/False are instances of int, hence check above
                negative_slope = param
            else:
                raise ValueError("negative_slope {} not a valid number".format(
                    param))
            return math.sqrt(2.0 / (1 + negative_slope**2))
        elif nonlinearity == 'selu':
            return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
        else:
            raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

    def _calculate_fan_in_and_fan_out(self, tensor):
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError(
                "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
            )

        num_input_fmaps = paddle.shape(tensor)[1]
        num_output_fmaps = paddle.shape(tensor)[0]
        receptive_field_size = 1
        if tensor.dim() > 2:
            for s in paddle.shape(tensor)[2:]:
                receptive_field_size *= s
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def _calculate_correct_fan(self, tensor, mode):
        mode = mode.lower()
        valid_modes = ['fan_in', 'fan_out']
        if mode not in valid_modes:
            raise ValueError("Mode {} not supported, please use one of {}".
                             format(mode, valid_modes))

        fan_in, fan_out = self._calculate_fan_in_and_fan_out(tensor)
        return fan_in if mode == 'fan_in' else fan_out

    def kaiming_normal_(self,
                        tensor,
                        a=0,
                        mode='fan_in',
                        nonlinearity='leaky_relu'):
        if 0 in paddle.shape(tensor):
            warnings.warn("Initializing zero-element tensors is a no-op.")
            return tensor
        fan = self._calculate_correct_fan(tensor, mode)
        gain = self.calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        with paddle.no_grad():
            initializer = paddle.nn.initializer.Normal(mean=0, std=std)
            initializer(tensor)

    def kaiming_uniform_(self,
                         tensor,
                         a=0,
                         mode='fan_in',
                         nonlinearity='leaky_relu'):
        if 0 in paddle.shape(tensor):
            warnings.warn("Initializing zero-element tensors is a no-op")
            return tensor
        fan = self._calculate_correct_fan(tensor, mode)
        gain = self.calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(
            3.0) * std  # Calculate uniform bounds from standard deviation
        with paddle.no_grad():
            initializer = paddle.nn.initializer.Uniform(-bound, bound)
        initializer(tensor)

    def constant_(self, tensor, a):
        with paddle.no_grad():
            initializer = paddle.nn.initializer.Constant(value=a)
            initializer(tensor)

    def kaiming_init(self,
                     module,
                     a=0,
                     mode='fan_out',
                     nonlinearity='relu',
                     bias=0,
                     distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            self.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            self.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, 'bias') and module.bias is not None:
            self.constant_(module.bias, bias)


class HRNetBasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2D(
            inplanes, planes, kernel_size=3, padding=1, bias_attr=False)
        self.bn1 = layers.SyncBatchNorm(planes)
        self.relu = nn.ReLU()
        self.deattn = PolarizedSelfAttentionModule_p(planes, planes)
        self.conv2 = nn.Conv2D(
            planes, planes, kernel_size=3, padding=1, bias_attr=False)
        self.bn2 = layers.SyncBatchNorm(planes)
        self.downsample = downsample

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
        super().__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = layers.SyncBatchNorm(planes)
        self.conv2 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias_attr=False)
        self.bn2 = layers.SyncBatchNorm(planes)
        self.conv3 = nn.Conv2D(
            planes, planes * self.expansion, kernel_size=1, bias_attr=False)
        self.bn3 = layers.SyncBatchNorm(planes * self.expansion)
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
    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 num_inchannels,
                 num_channels,
                 fuse_method,
                 multi_scale_output=True):
        super().__init__()
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != (num_channels[branch_index] *
                                                 block.expansion):
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False),
                layers.SyncBatchNorm(num_channels[branch_index] *
                                     block.expansion), )
        layer = []
        layer.append(
            block(self.num_inchannels[branch_index], num_channels[branch_index],
                  stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layer.append(
                block(self.num_inchannels[branch_index], num_channels[
                    branch_index]))
        return nn.Sequential(*layer)

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
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2D(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias_attr=False),
                            layers.SyncBatchNorm(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2D(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias_attr=False),
                                    layers.SyncBatchNorm(
                                        num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2D(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias_attr=False),
                                    layers.SyncBatchNorm(
                                        num_outchannels_conv3x3),
                                    nn.ReLU()))
                    fuse_layer.append(nn.Sequential(*conv3x3s))

            fuse_layers.append(nn.LayerList(fuse_layer))
        return nn.LayerList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

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
                    width_output = paddle.shape(x[i])[-1]
                    height_output = paddle.shape(x[i])[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=False)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {'BASIC': HRNetBasicBlock, 'BOTTLENECK': Bottleneck}


class HighResolutionNet(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 stage1_num_channels=[64],
                 stage1_num_blocks=[4],
                 stage2_num_channels=[48, 96],
                 stage2_num_modules=1,
                 stage2_num_branches=2,
                 stage2_num_blocks=[4, 4],
                 stage3_num_channels=[48, 96, 192],
                 stage3_num_modules=4,
                 stage3_num_branches=3,
                 stage3_num_blocks=[4, 4, 4],
                 stage4_num_channels=[48, 96, 192, 384],
                 stage4_num_modules=3,
                 stage4_num_branches=4,
                 stage4_num_blocks=[4, 4, 4, 4],
                 pretrained=None):
        super().__init__()

        self.stage1_num_channels = stage1_num_channels
        self.stage1_num_blocks = stage1_num_blocks
        self.stage2_num_channels = stage2_num_channels
        self.stage2_num_modules = stage2_num_modules
        self.stage2_num_branches = stage2_num_branches
        self.stage2_num_blocks = stage2_num_blocks
        self.stage3_num_channels = stage3_num_channels
        self.stage3_num_modules = stage3_num_modules
        self.stage3_num_branches = stage3_num_branches
        self.stage3_num_blocks = stage3_num_blocks
        self.stage4_num_channels = stage4_num_channels
        self.stage4_num_modules = stage4_num_modules
        self.stage4_num_branches = stage4_num_branches
        self.stage4_num_blocks = stage4_num_blocks
        self.pretrained = pretrained
        self.conv1 = nn.Conv2D(
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_attr=False)
        self.bn1 = layers.SyncBatchNorm(64)
        self.conv2 = nn.Conv2D(
            64, 64, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.bn2 = layers.SyncBatchNorm(64)
        self.relu = nn.ReLU()

        num_channels = self.stage1_num_channels[0]
        block = Bottleneck
        num_blocks = stage1_num_blocks[0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        num_channels = self.stage2_num_channels
        block = HRNetBasicBlock
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([stage1_out_channel],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_num_modules, self.stage2_num_branches,
            self.stage2_num_blocks, self.stage2_num_channels, num_channels,
            block)

        num_channels = self.stage3_num_channels
        block = HRNetBasicBlock
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_num_modules, self.stage3_num_branches,
            self.stage3_num_blocks, self.stage3_num_channels, num_channels,
            block)

        num_channels = self.stage4_num_channels
        block = HRNetBasicBlock
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_num_modules,
            self.stage4_num_branches,
            self.stage4_num_blocks,
            self.stage4_num_channels,
            num_channels,
            block,
            multi_scale_output=True)
        self.feat_channels = [np.int(np.sum(pre_stage_channels))]

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
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias_attr=False),
                            layers.SyncBatchNorm(num_channels_cur_layer[i]),
                            nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2D(
                                inchannels,
                                outchannels,
                                3,
                                2,
                                1,
                                bias_attr=False),
                            layers.SyncBatchNorm(outchannels),
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
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False),
                layers.SyncBatchNorm(planes * block.expansion), )

        layer = []
        layer.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layer.append(block(inplanes, planes))

        return nn.Sequential(*layer)

    def _make_stage(self,
                    num_modules,
                    num_branches,
                    num_blocks,
                    num_channels,
                    num_inchannels,
                    block,
                    fuse_method='sum',
                    multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches, block, num_blocks,
                                     num_inchannels, num_channels, fuse_method,
                                     reset_multi_scale_output))
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
        for i in range(self.stage2_num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_num_branches):
            if self.transition2[i] is not None:
                if i < self.stage2_num_branches:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                if i < self.stage3_num_branches:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        x0_h, x0_w = paddle.shape(x[0])[2:4]
        x1 = F.interpolate(
            x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(
            x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(
            x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=False)

        outs = [paddle.concat([x[0], x1, x2, x3], 1)]
        return outs

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


@manager.BACKBONES.add_component
def HRNetV2_PSA(**kwargs):
    model = HighResolutionNet(
        stage1_num_channels=[64],
        stage1_num_blocks=[4],
        stage2_num_channels=[48, 96],
        stage2_num_modules=1,
        stage2_num_branches=2,
        stage2_num_blocks=[4, 4],
        stage3_num_channels=[48, 96, 192],
        stage3_num_modules=4,
        stage3_num_branches=3,
        stage3_num_blocks=[4, 4, 4],
        stage4_num_channels=[48, 96, 192, 384],
        stage4_num_modules=3,
        stage4_num_branches=4,
        stage4_num_blocks=[4, 4, 4, 4],
        **kwargs)
    return model

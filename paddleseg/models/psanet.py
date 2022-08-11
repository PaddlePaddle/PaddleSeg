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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers
from paddleseg.utils import utils
from paddleseg.cvlibs import manager, param_init


class PolarizedSelfAttentionModule(nn.Layer):
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

        self.init_weight()

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)
        batch, channel, height, width = input_x.shape
        input_x = input_x.reshape((batch, channel, height * width))
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
        batch, channel, height, width = g_x.shape
        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.shape
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

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.kaiming_normal_init(layer.weight)


class HRNetBasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2D(
            inplanes, planes, kernel_size=3, padding=1, bias_attr=False)
        self.bn1 = layers.SyncBatchNorm(planes)
        self.relu = nn.ReLU()
        self.deattn = PolarizedSelfAttentionModule(planes, planes)
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
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
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
    def __init__(self, cfg_dic, pretrained=None):
        super().__init__()
        self.cfg_dic = cfg_dic
        self.conv1 = nn.Conv2D(
            3, 64, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.bn1 = layers.SyncBatchNorm(64)
        self.conv2 = nn.Conv2D(
            64, 64, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.bn2 = layers.SyncBatchNorm(64)
        self.relu = nn.ReLU()
        self.pretrained = pretrained

        self.stage1_cfg = self.cfg_dic['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = self.cfg_dic['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([stage1_out_channel],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg,
                                                           num_channels)

        self.stage3_cfg = self.cfg_dic['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg,
                                                           num_channels)

        self.stage4_cfg = self.cfg_dic['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion
            for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
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
        x0_h, x0_w = x[0].shape[2:4]
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


def HRNETV2_PSA(cfg_dic={
        'FINAL_CONV_KERNEL': 1,
        'STAGE1': {
            'NUM_MODULES': 1,
            'NUM_RANCHES': 1,
            'BLOCK': 'BOTTLENECK',
            'NUM_BLOCKS': [4],
            'NUM_CHANNELS': [64],
            'FUSE_METHOD': 'SUM'
        },
        'STAGE2': {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 2,
            'BLOCK': 'BASIC',
            'NUM_BLOCKS': [4, 4],
            'NUM_CHANNELS': [48, 96],
            'FUSE_METHOD': 'SUM'
        },
        'STAGE3': {
            'NUM_MODULES': 4,
            'NUM_BRANCHES': 3,
            'BLOCK': 'BASIC',
            'NUM_BLOCKS': [4, 4, 4],
            'NUM_CHANNELS': [48, 96, 192],
            'FUSE_METHOD': 'SUM'
        },
        'STAGE4': {
            'NUM_MODULES': 3,
            'NUM_BRANCHES': 4,
            'BLOCK': 'BASIC',
            'NUM_BLOCKS': [4, 4, 4, 4],
            'NUM_CHANNELS': [48, 96, 192, 384],
            'FUSE_METHOD': 'SUM'
        }
},
                pretrained=None):
    model = HighResolutionNet(cfg_dic=cfg_dic, pretrained=pretrained)
    return model


class AttenHead(nn.Layer):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        bot_ch = 256
        self.conv_bn_re0 = layers.ConvBNReLU(
            in_ch, bot_ch, kernel_size=3, padding=1, bias_attr=False)
        self.conv_bn_re1 = layers.ConvBNReLU(
            bot_ch, bot_ch, kernel_size=3, padding=1, bias_attr=False)
        self.conv2 = nn.Conv2D(bot_ch, out_ch, kernel_size=1, bias_attr=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_bn_re0(x)
        x = self.conv_bn_re1(x)
        x = self.conv2(x)
        x = self.sig(x)
        return x


class SpatialConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 padding='same',
                 **kwargs):
        super().__init__()

        self.conv_bn_relu_1 = layers.ConvBNReLU(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            **kwargs)

        self.conv_bn_relu_2 = layers.ConvBNReLU(
            out_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=padding,
            **kwargs)

    def forward(self, x):
        x = self.conv_bn_relu_1(x)
        x = self.conv_bn_relu_2(x)
        return x


class SpatialGatherModule(nn.Layer):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.

        Output:
          The correlation of every class map with every feature map
          shape = [n, num_feats, num_classes, 1]


    """

    def __init__(self, cls_num=0, scale=1):
        super().__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c = probs.shape[0], probs.shape[1]
        probs = probs.reshape((batch_size, c, -1))
        feats = feats.reshape((batch_size, feats.shape[1], -1))
        feats = feats.transpose((0, 2, 1))
        probs = F.softmax(self.scale * probs, axis=2)
        ocr_context = paddle.matmul(probs, feats)
        ocr_context = ocr_context.transpose((0, 2, 1)).unsqueeze(3)
        return ocr_context


class SpatialOCRModule(nn.Layer):
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2D(kernel_size=(scale, scale))
        self.f_pixel = SpatialConvBNReLU(
            self.in_channels,
            self.key_channels,
            kernel_size=1,
            padding=0,
            bias_attr=False)
        self.f_object = SpatialConvBNReLU(
            self.in_channels,
            self.key_channels,
            kernel_size=1,
            padding=0,
            bias_attr=False)
        self.f_down = layers.ConvBNReLU(
            self.in_channels,
            self.key_channels,
            kernel_size=1,
            padding=0,
            bias_attr=False)
        self.f_up = layers.ConvBNReLU(
            self.key_channels,
            self.in_channels,
            kernel_size=1,
            padding=0,
            bias_attr=False)

        _in_channels = 2 * in_channels
        self.conv_bn_dropout = nn.Sequential(
            layers.ConvBNReLU(
                _in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias_attr=False),
            nn.Dropout2D(dropout))

    def forward(self, feats, proxy):
        batch_size, _, h, w = feats.shape
        if self.scale > 1:
            feats = self.pool(feats)

        query = self.f_pixel(feats).reshape((batch_size, self.key_channels, -1))
        query = query.transpose((0, 2, 1))
        key = self.f_object(proxy).reshape((batch_size, self.key_channels, -1))
        value = self.f_down(proxy).reshape((batch_size, self.key_channels, -1))
        value = value.transpose((0, 2, 1))
        sim_map = paddle.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, axis=-1)
        context = paddle.matmul(sim_map, value)
        context = context.transpose((0, 2, 1))
        context = context.reshape((batch_size, self.key_channels,
                                   *feats.shape[2:]))
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(context, size=(h, w), mode='bilinear')
        output = paddle.concat([context, feats], 1)
        output = self.conv_bn_dropout(output)
        return output


class OCRHead(nn.Layer):
    def __init__(self, num_classes, in_channels):
        super().__init__()

        ocr_mid_channels = 512
        ocr_key_channels = 256
        self.indices = [-2, -1] if len(in_channels) > 1 else [-1, -1]
        high_level_ch = in_channels[self.indices[1]]
        self.conv3x3_ocr = layers.ConvBNReLU(
            high_level_ch, ocr_mid_channels, kernel_size=3, stride=1, padding=1)
        self.ocr_gather_head = SpatialGatherModule(num_classes)
        self.ocr_distri_head = SpatialOCRModule(
            in_channels=ocr_mid_channels,
            key_channels=ocr_key_channels,
            out_channels=ocr_mid_channels,
            scale=1,
            dropout=0.05, )
        self.cls_head = nn.Conv2D(
            ocr_mid_channels,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=True)
        self.aux_head = nn.Sequential(
            layers.ConvBNReLU(
                high_level_ch,
                high_level_ch,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.Conv2D(
                high_level_ch,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=True))
        self.init_weight()

    def forward(self, high_level_features):
        high_level_features = high_level_features[0]
        feats = self.conv3x3_ocr(high_level_features)
        aux_out = self.aux_head(high_level_features)
        context = self.ocr_gather_head(feats, aux_out)
        ocr_feats = self.ocr_distri_head(feats, context)
        cls_out = self.cls_head(ocr_feats)
        return cls_out, aux_out, ocr_feats

    def init_weight(self):
        """Initialize the parameters of model parts."""
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                param_init.normal_init(sublayer.weight, std=0.001)
            elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(sublayer.weight, value=1.0)
                param_init.constant_init(sublayer.bias, value=0.0)


@manager.MODELS.add_component
class PSANet(nn.Layer):
    """
    The PSA implementation based on PaddlePaddle.

    The original article refers to
    Huajun Liu, Fuqiang Liu et al. "Polarized Self-Attention"
    (https://arxiv.org/pdf/2107.00782.pdf).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network, currently support HRNETV2_PSA.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone=HRNETV2_PSA(),
                 backbone_indices=[0],
                 mscale=[0.5, 1.0, 2.0],
                 pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.backbone_indices = backbone_indices
        self.mscale = mscale
        in_channels = [self.backbone.feat_channels[i] for i in backbone_indices]
        self.ocr = OCRHead(num_classes, in_channels)
        self.scale_attn = AttenHead(in_ch=512, out_ch=1)
        self.init_weight()

    def _fwd(self, x):
        x_size = x.shape[2:]
        high_level_features = self.backbone(x)
        cls_out, aux_out, ocr_mid_feats = self.ocr(high_level_features)
        attn = self.scale_attn(ocr_mid_feats)
        aux_out = F.interpolate(aux_out, size=x_size, mode='bilinear')
        cls_out = F.interpolate(cls_out, size=x_size, mode='bilinear')
        attn = F.interpolate(attn, size=x_size, mode='bilinear')

        return {'cls_out': cls_out, 'aux_out': aux_out, 'logit_attn': attn}

    def nscale_forward(self, inputs, scales):
        x_1x = inputs
        scales = sorted(scales, reverse=True)
        pred = None
        aux = None
        output_dict = {}
        for s in scales:
            x = F.interpolate(x_1x, scale_factor=s, mode='bilinear')
            outs = self._fwd(x)
            cls_out = outs['cls_out']
            attn_out = outs['logit_attn']
            aux_out = outs['aux_out']
            key_pred = 'pred_' + str(float(s)).replace('.', '') + 'x'
            output_dict[key_pred] = cls_out
            if s != 2.0:
                key_attn = 'attn_' + str(float(s)).replace('.', '') + 'x'
                output_dict[key_attn] = attn_out
            if pred is None:
                pred = cls_out
                aux = aux_out
            elif s >= 1.0:
                pred = F.interpolate(
                    pred, size=cls_out.shape[2:4], mode='bilinear')
                pred = attn_out * cls_out + (1 - attn_out) * pred
                aux = F.interpolate(
                    aux, size=cls_out.shape[2:4], mode='bilinear')
                aux = attn_out * aux_out + (1 - attn_out) * aux
            else:
                cls_out = attn_out * cls_out
                aux_out = attn_out * aux_out
                cls_out = F.interpolate(
                    cls_out, size=pred.shape[2:4], mode='bilinear')
                aux_out = F.interpolate(
                    aux_out, size=pred.shape[2:4], mode='bilinear')
                attn_out = F.interpolate(
                    attn_out, size=pred.shape[2:4], mode='bilinear')
                pred = cls_out + (1 - attn_out) * pred
                aux = aux_out + (1 - attn_out) * aux
        logit_list = [aux, pred] if self.training else [pred]
        return logit_list

    def two_scale_forward(self, inputs):
        x_lo = F.interpolate(inputs, scale_factor=0.5, mode='bilinear')
        lo_outs = self._fwd(x_lo)
        pred_05x = lo_outs['cls_out']
        p_lo = pred_05x
        aux_lo = lo_outs['aux_out']
        logit_attn = lo_outs['logit_attn']

        hi_outs = self._fwd(inputs)
        pred_10x = hi_outs['cls_out']
        p_1x = pred_10x
        aux_1x = hi_outs['aux_out']

        p_lo = logit_attn * p_lo
        aux_lo = logit_attn * aux_lo
        p_lo = F.interpolate(p_lo, size=p_1x.shape[2:4], mode='bilinear')

        aux_lo = F.interpolate(aux_lo, size=p_1x.shape[2:4], mode='bilinear')

        logit_attn = F.interpolate(
            logit_attn, size=p_1x.shape[2:4], mode='bilinear')

        joint_pred = p_lo + (1 - logit_attn) * p_1x
        joint_aux = aux_lo + (1 - logit_attn) * aux_1x
        if self.training:
            scaled_pred_05x = F.interpolate(
                pred_05x, size=p_1x.shape[2:4], mode='bilinear')
            logit_list = [joint_aux, joint_pred, scaled_pred_05x, pred_10x]
        else:
            logit_list = [joint_pred]
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, inputs):
        if self.mscale and not self.training:
            return self.nscale_forward(inputs, self.mscale)
        return self.two_scale_forward(inputs)

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers


@manager.MODELS.add_component
class ESPNetV2(nn.Layer):
    """
    The ESPNetV2 implementation based on PaddlePaddle.

    The original article refers to
    Sachin Mehta, Mohammad Rastegari, Linda Shapiro, and Hannaneh Hajishirzi. "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network"
    (https://arxiv.org/abs/1811.11431).

    Args:
        num_classes (int): The unique number of target classes.
        in_channels (int, optional): Number of input channels. Default: 3.
        scale (float, optional): The scale of channels, only support scale <= 1.5 and scale == 2. Default: 1.0.
        drop_prob (floa, optional): The probability of dropout. Default: 0.1.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 in_channels=3,
                 scale=1.0,
                 drop_prob=0.1,
                 pretrained=None):
        super().__init__()
        self.backbone = EESPNetBackbone(in_channels, drop_prob, scale)
        self.in_channels = self.backbone.out_channels
        self.proj_l4_c = layers.ConvBNPReLU(
            self.in_channels[3],
            self.in_channels[2],
            1,
            stride=1,
            bias_attr=False)
        psp_size = 2 * self.in_channels[2]
        self.eesp_psp = nn.Sequential(
            EESP(
                psp_size,
                psp_size // 2,
                stride=1,
                branches=4,
                kernel_size_maximum=7),
            PSPModule(psp_size // 2, psp_size // 2), )

        self.project_l3 = nn.Sequential(
            nn.Dropout2D(p=drop_prob),
            nn.Conv2D(
                psp_size // 2, num_classes, 1, 1, bias_attr=False), )
        self.act_l3 = BNPReLU(num_classes)
        self.project_l2 = layers.ConvBNPReLU(
            self.in_channels[1] + num_classes,
            num_classes,
            1,
            stride=1,
            bias_attr=False)
        self.project_l1 = nn.Sequential(
            nn.Dropout2D(p=drop_prob),
            nn.Conv2D(
                self.in_channels[0] + num_classes,
                num_classes,
                1,
                1,
                bias_attr=False), )

        self.pretrained = pretrained

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def hierarchical_upsample(self, x, factor=3):
        for i in range(factor):
            x = F.interpolate(
                x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        out_l1, out_l2, out_l3, out_l4 = self.backbone(x)

        out_l4_proj = self.proj_l4_c(out_l4)
        l4_to_l3 = F.interpolate(
            out_l4_proj, scale_factor=2, mode='bilinear', align_corners=True)
        merged_l3 = self.eesp_psp(paddle.concat([out_l3, l4_to_l3], axis=1))
        proj_merge_l3 = self.project_l3(merged_l3)
        proj_merge_l3 = self.act_l3(proj_merge_l3)

        l3_to_l2 = F.interpolate(
            proj_merge_l3, scale_factor=2, mode='bilinear', align_corners=True)
        merged_l2 = self.project_l2(paddle.concat([out_l2, l3_to_l2], axis=1))

        l2_to_l1 = F.interpolate(
            merged_l2, scale_factor=2, mode='bilinear', align_corners=True)
        merged_l1 = self.project_l1(paddle.concat([out_l1, l2_to_l1], axis=1))

        if self.training:
            return [
                F.interpolate(
                    merged_l1,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True),
                self.hierarchical_upsample(proj_merge_l3),
            ]
        else:
            return [
                F.interpolate(
                    merged_l1,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True)
            ]


class BNPReLU(nn.Layer):
    def __init__(self, out_channels, **kwargs):
        super().__init__()
        if 'data_format' in kwargs:
            data_format = kwargs['data_format']
        else:
            data_format = 'NCHW'
        self._batch_norm = layers.SyncBatchNorm(
            out_channels, data_format=data_format)
        self._prelu = layers.Activation("prelu")

    def forward(self, x):
        x = self._batch_norm(x)
        x = self._prelu(x)
        return x


class EESP(nn.Layer):
    """
    EESP block, principle: reduce -> split -> transform -> merge

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2. Default: 1.
        branches (int, optional): Number of branches. Default: 4.
        kernel_size_maximum (int, optional): A maximum value of receptive field allowed for EESP block. Default: 7.
        down_method (str, optional): Down sample or not, only support 'avg' and 'esp'(equivalent to stride is 2 or not). Default: 'esp'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 branches=4,
                 kernel_size_maximum=7,
                 down_method='esp'):
        super(EESP, self).__init__()
        if out_channels % branches != 0:
            raise RuntimeError(
                "The out_channes for EESP should be factorized by branches, but out_channels={} cann't be factorized by branches={}"
                .format(out_channels, branches))
        assert down_method in [
            'avg', 'esp'
        ], "The down_method for EESP only support 'avg' or 'esp', but got down_method={}".format(
            down_method)
        self.in_channels = in_channels
        self.stride = stride

        in_branch_channels = int(out_channels / branches)
        self.group_conv_in = layers.ConvBNPReLU(
            in_channels,
            in_branch_channels,
            1,
            stride=1,
            groups=branches,
            bias_attr=False)

        map_ksize_dilation = {
            3: 1,
            5: 2,
            7: 3,
            9: 4,
            11: 5,
            13: 6,
            15: 7,
            17: 8
        }
        self.kernel_sizes = []
        for i in range(branches):
            kernel_size = 3 + 2 * i
            kernel_size = kernel_size if kernel_size <= kernel_size_maximum else 3
            self.kernel_sizes.append(kernel_size)
        self.kernel_sizes.sort()

        self.spp_modules = nn.LayerList()
        for i in range(branches):
            dilation = map_ksize_dilation[self.kernel_sizes[i]]
            self.spp_modules.append(
                nn.Conv2D(
                    in_branch_channels,
                    in_branch_channels,
                    kernel_size=3,
                    padding='same',
                    stride=stride,
                    dilation=dilation,
                    groups=in_branch_channels,
                    bias_attr=False))
        self.group_conv_out = layers.ConvBN(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            groups=branches,
            bias_attr=False)
        self.bn_act = BNPReLU(out_channels)
        self._act = nn.PReLU()
        self.down_method = True if down_method == 'avg' else False

    @paddle.jit.not_to_static
    def convert_group_x(self, group_merge, x):
        if x.shape == group_merge.shape:
            group_merge += x

        return group_merge

    def forward(self, x):
        group_out = self.group_conv_in(x)
        output = [self.spp_modules[0](group_out)]

        for k in range(1, len(self.spp_modules)):
            output_k = self.spp_modules[k](group_out)
            output_k = output_k + output[k - 1]
            output.append(output_k)

        group_merge = self.group_conv_out(
            self.bn_act(paddle.concat(
                output, axis=1)))

        if self.stride == 2 and self.down_method:
            return group_merge

        group_merge = self.convert_group_x(group_merge, x)
        out = self._act(group_merge)
        return out


class PSPModule(nn.Layer):
    def __init__(self, in_channels, out_channels, sizes=4):
        super().__init__()
        self.stages = nn.LayerList([
            nn.Conv2D(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                groups=in_channels,
                padding='same',
                bias_attr=False) for _ in range(sizes)
        ])
        self.project = layers.ConvBNPReLU(
            in_channels * (sizes + 1),
            out_channels,
            1,
            stride=1,
            bias_attr=False)

    def forward(self, feats):
        h, w = paddle.shape(feats)[2:4]
        out = [feats]
        for stage in self.stages:
            feats = F.avg_pool2d(feats, kernel_size=3, stride=2, padding='same')
            upsampled = F.interpolate(
                stage(feats), size=[h, w], mode='bilinear', align_corners=True)
            out.append(upsampled)
        return self.project(paddle.concat(out, axis=1))


class DownSampler(nn.Layer):
    """
    Down sampler.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        branches (int, optional): Number of branches. Default: 9.
        kernel_size_maximum (int, optional): A maximum value of kernel_size for EESP block. Default: 9.
        shortcut (bool, optional): Use shortcut or not. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 branches=4,
                 kernel_size_maximum=9,
                 shortcut=True):
        super().__init__()
        if out_channels < in_channels:
            raise RuntimeError(
                "The out_channes for DownSampler should be bigger than in_channels, but got in_channles={}, out_channels={}"
                .format(in_channels, out_channels))
        self.eesp = EESP(
            in_channels,
            out_channels - in_channels,
            stride=2,
            branches=branches,
            kernel_size_maximum=kernel_size_maximum,
            down_method='avg')
        self.avg = nn.AvgPool2D(kernel_size=3, padding=1, stride=2)
        if shortcut:
            self.shortcut_layer = nn.Sequential(
                layers.ConvBNPReLU(
                    3, 3, 3, stride=1, bias_attr=False),
                layers.ConvBN(
                    3, out_channels, 1, stride=1, bias_attr=False), )
        self._act = nn.PReLU()

    def forward(self, x, inputs=None):
        avg_out = self.avg(x)
        eesp_out = self.eesp(x)
        output = paddle.concat([avg_out, eesp_out], axis=1)

        if inputs is not None:
            w1 = paddle.shape(avg_out)[2]
            w2 = paddle.shape(inputs)[2]

            while w2 != w1:
                inputs = F.avg_pool2d(
                    inputs, kernel_size=3, padding=1, stride=2)
                w2 = paddle.shape(inputs)[2]
            # import pdb
            # pdb.set_trace()
            output = output + self.shortcut_layer(inputs)
        return self._act(output)


class EESPNetBackbone(nn.Layer):
    """
    The EESPNetBackbone implementation based on PaddlePaddle.

    The original article refers to
    Sachin Mehta, Mohammad Rastegari, Linda Shapiro, and Hannaneh Hajishirzi. "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network"
    (https://arxiv.org/abs/1811.11431).

    Args:
        in_channels (int, optional): Number of input channels. Default: 3.
        drop_prob (float, optional): The probability of dropout. Default: 3.
        scale (float, optional): The scale of channels, only support scale <= 1.5 and scale == 2. Default: 1.0.
    """

    def __init__(self, in_channels=3, drop_prob=0.1, scale=1.0):
        super().__init__()
        reps = [0, 3, 7, 3]

        num_level = 4  # 1/2, 1/4, 1/8, 1/16
        kernel_size_limitations = [13, 11, 9, 7]  # kernel size limitation
        branch_list = [4] * len(
            kernel_size_limitations)  # branches at different levels

        base_channels = 32  # first conv output channels
        channels_config = [base_channels] * num_level

        for i in range(num_level):
            if i == 0:
                channels = int(base_channels * scale)
                channels = math.ceil(channels / branch_list[0]) * branch_list[0]
                channels_config[
                    i] = base_channels if channels > base_channels else channels
            else:
                channels_config[i] = channels * pow(2, i)

        self.level1 = layers.ConvBNPReLU(
            in_channels, channels_config[0], 3, stride=2, bias_attr=False)

        self.level2 = DownSampler(
            channels_config[0],
            channels_config[1],
            branches=branch_list[0],
            kernel_size_maximum=kernel_size_limitations[0],
            shortcut=True)

        self.level3_0 = DownSampler(
            channels_config[1],
            channels_config[2],
            branches=branch_list[1],
            kernel_size_maximum=kernel_size_limitations[1],
            shortcut=True)
        self.level3 = nn.LayerList()
        for i in range(reps[1]):
            self.level3.append(
                EESP(
                    channels_config[2],
                    channels_config[2],
                    stride=1,
                    branches=branch_list[2],
                    kernel_size_maximum=kernel_size_limitations[2]))

        self.level4_0 = DownSampler(
            channels_config[2],
            channels_config[3],
            branches=branch_list[2],
            kernel_size_maximum=kernel_size_limitations[2],
            shortcut=True)
        self.level4 = nn.LayerList()
        for i in range(reps[2]):
            self.level4.append(
                EESP(
                    channels_config[3],
                    channels_config[3],
                    stride=1,
                    branches=branch_list[3],
                    kernel_size_maximum=kernel_size_limitations[3]))

        self.out_channels = channels_config

        self.init_params()

    def init_params(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                param_init.kaiming_normal_init(m.weight)
                if m.bias is not None:
                    param_init.constant_init(m.bias, value=0.0)
            elif isinstance(m, nn.BatchNorm2D):
                param_init.constant_init(m.weight, value=1.0)
                param_init.constant_init(m.bias, value=0.0)
            elif isinstance(m, nn.Linear):
                param_init.normal_init(m.weight, std=0.001)
                if m.bias is not None:
                    param_init.constant_init(m.bias, value=0.0)

    def forward(self, x):
        out_l1 = self.level1(x)
        out_l2 = self.level2(out_l1, x)
        out_l3 = self.level3_0(out_l2, x)
        for i, layer in enumerate(self.level3):
            out_l3 = layer(out_l3)
        out_l4 = self.level4_0(out_l3, x)
        for i, layer in enumerate(self.level4):
            out_l4 = layer(out_l4)
        return out_l1, out_l2, out_l3, out_l4


if __name__ == '__main__':
    import paddle
    import numpy as np

    paddle.enable_static()

    startup_prog = paddle.static.default_startup_program()

    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(startup_prog)
    path_prefix = "./output/model"

    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(path_prefix, exe))
    print('inference_program:', inference_program)

    tensor_img = np.array(
        np.random.random((1, 3, 1024, 2048)), dtype=np.float32)
    results = exe.run(inference_program,
                      feed={feed_target_names[0]: tensor_img},
                      fetch_list=fetch_targets)

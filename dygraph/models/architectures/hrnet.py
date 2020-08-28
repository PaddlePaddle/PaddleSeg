# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from paddle.fluid.initializer import Normal
from paddle.nn import SyncBatchNorm as BatchNorm

from dygraph.cvlibs import manager

__all__ = [
    "HRNet_W18_Small_V1", "HRNet_W18_Small_V2", "HRNet_W18", "HRNet_W30",
    "HRNet_W32", "HRNet_W40", "HRNet_W44", "HRNet_W48", "HRNet_W60", "HRNet_W64"
]


class HRNet(fluid.dygraph.Layer):
    """
    HRNet：Deep High-Resolution Representation Learning for Visual Recognition
    https://arxiv.org/pdf/1908.07919.pdf.

    Args:
        stage1_num_modules (int): number of modules for stage1. Default 1.
        stage1_num_blocks (list): number of blocks per module for stage1. Default [4].
        stage1_num_channels (list): number of channels per branch for stage1. Default [64].
        stage2_num_modules (int): number of modules for stage2. Default 1.
        stage2_num_blocks (list): number of blocks per module for stage2. Default [4, 4]
        stage2_num_channels (list): number of channels per branch for stage2. Default [18, 36].
        stage3_num_modules (int): number of modules for stage3. Default 4.
        stage3_num_blocks (list): number of blocks per module for stage3. Default [4, 4, 4]
        stage3_num_channels (list): number of channels per branch for stage3. Default [18, 36, 72].
        stage4_num_modules (int): number of modules for stage4. Default 3.
        stage4_num_blocks (list): number of blocks per module for stage4. Default [4, 4, 4, 4]
        stage4_num_channels (list): number of channels per branch for stage4. Default [18, 36, 72. 144].
        has_se (bool): whether to use Squeeze-and-Excitation module. Default False.
    """

    def __init__(self,
                 stage1_num_modules=1,
                 stage1_num_blocks=[4],
                 stage1_num_channels=[64],
                 stage2_num_modules=1,
                 stage2_num_blocks=[4, 4],
                 stage2_num_channels=[18, 36],
                 stage3_num_modules=4,
                 stage3_num_blocks=[4, 4, 4],
                 stage3_num_channels=[18, 36, 72],
                 stage4_num_modules=3,
                 stage4_num_blocks=[4, 4, 4, 4],
                 stage4_num_channels=[18, 36, 72, 144],
                 has_se=False):
        super(HRNet, self).__init__()

        self.stage1_num_modules = stage1_num_modules
        self.stage1_num_blocks = stage1_num_blocks
        self.stage1_num_channels = stage1_num_channels
        self.stage2_num_modules = stage2_num_modules
        self.stage2_num_blocks = stage2_num_blocks
        self.stage2_num_channels = stage2_num_channels
        self.stage3_num_modules = stage3_num_modules
        self.stage3_num_blocks = stage3_num_blocks
        self.stage3_num_channels = stage3_num_channels
        self.stage4_num_modules = stage4_num_modules
        self.stage4_num_blocks = stage4_num_blocks
        self.stage4_num_channels = stage4_num_channels
        self.has_se = has_se

        self.conv_layer1_1 = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=3,
            stride=2,
            act='relu',
            name="layer1_1")

        self.conv_layer1_2 = ConvBNLayer(
            num_channels=64,
            num_filters=64,
            filter_size=3,
            stride=2,
            act='relu',
            name="layer1_2")

        self.la1 = Layer1(
            num_channels=64,
            num_blocks=self.stage1_num_blocks[0],
            num_filters=self.stage1_num_channels[0],
            has_se=has_se,
            name="layer2")

        self.tr1 = TransitionLayer(
            in_channels=[self.stage1_num_channels[0] * 4],
            out_channels=self.stage2_num_channels,
            name="tr1")

        self.st2 = Stage(
            num_channels=self.stage2_num_channels,
            num_modules=self.stage2_num_modules,
            num_blocks=self.stage2_num_blocks,
            num_filters=self.stage2_num_channels,
            has_se=self.has_se,
            name="st2")

        self.tr2 = TransitionLayer(
            in_channels=self.stage2_num_channels,
            out_channels=self.stage3_num_channels,
            name="tr2")
        self.st3 = Stage(
            num_channels=self.stage3_num_channels,
            num_modules=self.stage3_num_modules,
            num_blocks=self.stage3_num_blocks,
            num_filters=self.stage3_num_channels,
            has_se=self.has_se,
            name="st3")

        self.tr3 = TransitionLayer(
            in_channels=self.stage3_num_channels,
            out_channels=self.stage4_num_channels,
            name="tr3")
        self.st4 = Stage(
            num_channels=self.stage4_num_channels,
            num_modules=self.stage4_num_modules,
            num_blocks=self.stage4_num_blocks,
            num_filters=self.stage4_num_channels,
            has_se=self.has_se,
            name="st4")

    def forward(self, x, label=None, mode='train'):
        input_shape = x.shape[2:]
        conv1 = self.conv_layer1_1(x)
        conv2 = self.conv_layer1_2(conv1)

        la1 = self.la1(conv2)

        tr1 = self.tr1([la1])
        st2 = self.st2(tr1)

        tr2 = self.tr2(st2)
        st3 = self.st3(tr2)

        tr3 = self.tr3(st3)
        st4 = self.st4(tr3)

        x0_h, x0_w = st4[0].shape[2:]
        x1 = fluid.layers.resize_bilinear(st4[1], out_shape=(x0_h, x0_w))
        x2 = fluid.layers.resize_bilinear(st4[2], out_shape=(x0_h, x0_w))
        x3 = fluid.layers.resize_bilinear(st4[3], out_shape=(x0_h, x0_w))
        x = fluid.layers.concat([st4[0], x1, x2, x3], axis=1)

        return x


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act="relu",
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            param_attr=ParamAttr(
                initializer=Normal(scale=0.001), name=name + "_weights"),
            bias_attr=False)
        bn_name = name + '_bn'
        self._batch_norm = BatchNorm(
            num_filters,
            weight_attr=ParamAttr(
                name=bn_name + '_scale',
                initializer=fluid.initializer.Constant(1.0)),
            bias_attr=ParamAttr(
                bn_name + '_offset',
                initializer=fluid.initializer.Constant(0.0)))
        self.act = act

    def forward(self, input):
        y = self._conv(input)
        y = self._batch_norm(y)
        if self.act == 'relu':
            y = fluid.layers.relu(y)
        return y


class Layer1(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 num_blocks,
                 has_se=False,
                 name=None):
        super(Layer1, self).__init__()

        self.bottleneck_block_list = []

        for i in range(num_blocks):
            bottleneck_block = self.add_sublayer(
                "bb_{}_{}".format(name, i + 1),
                BottleneckBlock(
                    num_channels=num_channels if i == 0 else num_filters * 4,
                    num_filters=num_filters,
                    has_se=has_se,
                    stride=1,
                    downsample=True if i == 0 else False,
                    name=name + '_' + str(i + 1)))
            self.bottleneck_block_list.append(bottleneck_block)

    def forward(self, input):
        conv = input
        for block_func in self.bottleneck_block_list:
            conv = block_func(conv)
        return conv


class TransitionLayer(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, name=None):
        super(TransitionLayer, self).__init__()

        num_in = len(in_channels)
        num_out = len(out_channels)
        self.conv_bn_func_list = []
        for i in range(num_out):
            residual = None
            if i < num_in:
                if in_channels[i] != out_channels[i]:
                    residual = self.add_sublayer(
                        "transition_{}_layer_{}".format(name, i + 1),
                        ConvBNLayer(
                            num_channels=in_channels[i],
                            num_filters=out_channels[i],
                            filter_size=3,
                            name=name + '_layer_' + str(i + 1)))
            else:
                residual = self.add_sublayer(
                    "transition_{}_layer_{}".format(name, i + 1),
                    ConvBNLayer(
                        num_channels=in_channels[-1],
                        num_filters=out_channels[i],
                        filter_size=3,
                        stride=2,
                        name=name + '_layer_' + str(i + 1)))
            self.conv_bn_func_list.append(residual)

    def forward(self, input):
        outs = []
        for idx, conv_bn_func in enumerate(self.conv_bn_func_list):
            if conv_bn_func is None:
                outs.append(input[idx])
            else:
                if idx < len(input):
                    outs.append(conv_bn_func(input[idx]))
                else:
                    outs.append(conv_bn_func(input[-1]))
        return outs


class Branches(fluid.dygraph.Layer):
    def __init__(self,
                 num_blocks,
                 in_channels,
                 out_channels,
                 has_se=False,
                 name=None):
        super(Branches, self).__init__()

        self.basic_block_list = []

        for i in range(len(out_channels)):
            self.basic_block_list.append([])
            for j in range(num_blocks[i]):
                in_ch = in_channels[i] if j == 0 else out_channels[i]
                basic_block_func = self.add_sublayer(
                    "bb_{}_branch_layer_{}_{}".format(name, i + 1, j + 1),
                    BasicBlock(
                        num_channels=in_ch,
                        num_filters=out_channels[i],
                        has_se=has_se,
                        name=name + '_branch_layer_' + str(i + 1) + '_' +
                        str(j + 1)))
                self.basic_block_list[i].append(basic_block_func)

    def forward(self, inputs):
        outs = []
        for idx, input in enumerate(inputs):
            conv = input
            for basic_block_func in self.basic_block_list[idx]:
                conv = basic_block_func(conv)
            outs.append(conv)
        return outs


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se,
                 stride=1,
                 downsample=False,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act="relu",
            name=name + "_conv1",
        )
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="relu",
            name=name + "_conv2")
        self.conv3 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_conv3")

        if self.downsample:
            self.conv_down = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                act=None,
                name=name + "_downsample")

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters * 4,
                num_filters=num_filters * 4,
                reduction_ratio=16,
                name=name + '_fc')

    def forward(self, input):
        residual = input
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        if self.downsample:
            residual = self.conv_down(input)

        if self.has_se:
            conv3 = self.se(conv3)

        y = fluid.layers.elementwise_add(x=conv3, y=residual, act="relu")
        return y


class BasicBlock(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride=1,
                 has_se=False,
                 downsample=False,
                 name=None):
        super(BasicBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="relu",
            name=name + "_conv1")
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=1,
            act=None,
            name=name + "_conv2")

        if self.downsample:
            self.conv_down = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                act="relu",
                name=name + "_downsample")

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters,
                num_filters=num_filters,
                reduction_ratio=16,
                name=name + '_fc')

    def forward(self, input):
        residual = input
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)

        if self.downsample:
            residual = self.conv_down(input)

        if self.has_se:
            conv2 = self.se(conv2)

        y = fluid.layers.elementwise_add(x=conv2, y=residual, act="relu")
        return y


class SELayer(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, reduction_ratio, name=None):
        super(SELayer, self).__init__()

        self.pool2d_gap = Pool2D(pool_type='avg', global_pooling=True)

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = Linear(
            num_channels,
            med_ch,
            act="relu",
            param_attr=ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + "_sqz_weights"),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))

        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = Linear(
            med_ch,
            num_filters,
            act="sigmoid",
            param_attr=ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + "_exc_weights"),
            bias_attr=ParamAttr(name=name + '_exc_offset'))

    def forward(self, input):
        pool = self.pool2d_gap(input)
        pool = fluid.layers.reshape(pool, shape=[-1, self._num_channels])
        squeeze = self.squeeze(pool)
        excitation = self.excitation(squeeze)
        excitation = fluid.layers.reshape(
            excitation, shape=[-1, self._num_channels, 1, 1])
        out = input * excitation
        return out


class Stage(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_modules,
                 num_blocks,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 name=None):
        super(Stage, self).__init__()

        self._num_modules = num_modules

        self.stage_func_list = []
        for i in range(num_modules):
            if i == num_modules - 1 and not multi_scale_output:
                stage_func = self.add_sublayer(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                        has_se=has_se,
                        multi_scale_output=False,
                        name=name + '_' + str(i + 1)))
            else:
                stage_func = self.add_sublayer(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                        has_se=has_se,
                        name=name + '_' + str(i + 1)))

            self.stage_func_list.append(stage_func)

    def forward(self, input):
        out = input
        for idx in range(self._num_modules):
            out = self.stage_func_list[idx](out)
        return out


class HighResolutionModule(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_blocks,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 name=None):
        super(HighResolutionModule, self).__init__()

        self.branches_func = Branches(
            num_blocks=num_blocks,
            in_channels=num_channels,
            out_channels=num_filters,
            has_se=has_se,
            name=name)

        self.fuse_func = FuseLayers(
            in_channels=num_filters,
            out_channels=num_filters,
            multi_scale_output=multi_scale_output,
            name=name)

    def forward(self, input):
        out = self.branches_func(input)
        out = self.fuse_func(out)
        return out


class FuseLayers(fluid.dygraph.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 multi_scale_output=True,
                 name=None):
        super(FuseLayers, self).__init__()

        self._actual_ch = len(in_channels) if multi_scale_output else 1
        self._in_channels = in_channels

        self.residual_func_list = []
        for i in range(self._actual_ch):
            for j in range(len(in_channels)):
                residual_func = None
                if j > i:
                    residual_func = self.add_sublayer(
                        "residual_{}_layer_{}_{}".format(name, i + 1, j + 1),
                        ConvBNLayer(
                            num_channels=in_channels[j],
                            num_filters=out_channels[i],
                            filter_size=1,
                            stride=1,
                            act=None,
                            name=name + '_layer_' + str(i + 1) + '_' +
                            str(j + 1)))
                    self.residual_func_list.append(residual_func)
                elif j < i:
                    pre_num_filters = in_channels[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            residual_func = self.add_sublayer(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvBNLayer(
                                    num_channels=pre_num_filters,
                                    num_filters=out_channels[i],
                                    filter_size=3,
                                    stride=2,
                                    act=None,
                                    name=name + '_layer_' + str(i + 1) + '_' +
                                    str(j + 1) + '_' + str(k + 1)))
                            pre_num_filters = out_channels[i]
                        else:
                            residual_func = self.add_sublayer(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvBNLayer(
                                    num_channels=pre_num_filters,
                                    num_filters=out_channels[j],
                                    filter_size=3,
                                    stride=2,
                                    act="relu",
                                    name=name + '_layer_' + str(i + 1) + '_' +
                                    str(j + 1) + '_' + str(k + 1)))
                            pre_num_filters = out_channels[j]
                        self.residual_func_list.append(residual_func)

    def forward(self, input):
        outs = []
        residual_func_idx = 0
        for i in range(self._actual_ch):
            residual = input[i]
            residual_shape = residual.shape[-2:]
            for j in range(len(self._in_channels)):
                if j > i:
                    y = self.residual_func_list[residual_func_idx](input[j])
                    residual_func_idx += 1

                    y = fluid.layers.resize_bilinear(
                        input=y, out_shape=residual_shape)
                    residual = fluid.layers.elementwise_add(
                        x=residual, y=y, act=None)
                elif j < i:
                    y = input[j]
                    for k in range(i - j):
                        y = self.residual_func_list[residual_func_idx](y)
                        residual_func_idx += 1

                    residual = fluid.layers.elementwise_add(
                        x=residual, y=y, act=None)

            layer_helper = LayerHelper(self.full_name(), act='relu')
            residual = layer_helper.append_activation(residual)
            outs.append(residual)

        return outs


class LastClsOut(fluid.dygraph.Layer):
    def __init__(self,
                 num_channel_list,
                 has_se,
                 num_filters_list=[32, 64, 128, 256],
                 name=None):
        super(LastClsOut, self).__init__()

        self.func_list = []
        for idx in range(len(num_channel_list)):
            func = self.add_sublayer(
                "conv_{}_conv_{}".format(name, idx + 1),
                BottleneckBlock(
                    num_channels=num_channel_list[idx],
                    num_filters=num_filters_list[idx],
                    has_se=has_se,
                    downsample=True,
                    name=name + 'conv_' + str(idx + 1)))
            self.func_list.append(func)

    def forward(self, inputs):
        outs = []
        for idx, input in enumerate(inputs):
            out = self.func_list[idx](input)
            outs.append(out)
        return outs


@manager.BACKBONES.add_component
def HRNet_W18_Small_V1(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[1],
        stage1_num_channels=[32],
        stage2_num_modules=1,
        stage2_num_blocks=[2, 2],
        stage2_num_channels=[16, 32],
        stage3_num_modules=1,
        stage3_num_blocks=[2, 2, 2],
        stage3_num_channels=[16, 32, 64],
        stage4_num_modules=1,
        stage4_num_blocks=[2, 2, 2, 2],
        stage4_num_channels=[16, 32, 64, 128],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W18_Small_V2(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[2],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[2, 2],
        stage2_num_channels=[18, 36],
        stage3_num_modules=1,
        stage3_num_blocks=[2, 2, 2],
        stage3_num_channels=[18, 36, 72],
        stage4_num_modules=1,
        stage4_num_blocks=[2, 2, 2, 2],
        stage4_num_channels=[18, 36, 72, 144],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W18(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[18, 36],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[18, 36, 72],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[18, 36, 72, 144],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W30(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[30, 60],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[30, 60, 120],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[30, 60, 120, 240],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W32(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[32, 64],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[32, 64, 128],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[32, 64, 128, 256],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W40(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[40, 80],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[40, 80, 160],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[40, 80, 160, 320],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W44(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[44, 88],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[44, 88, 176],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[44, 88, 176, 352],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W48(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[48, 96],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[48, 96, 192],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[48, 96, 192, 384],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W60(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[60, 120],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[60, 120, 240],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[60, 120, 240, 480],
        **kwargs)
    return model


@manager.BACKBONES.add_component
def HRNet_W64(**kwargs):
    model = HRNet(
        stage1_num_modules=1,
        stage1_num_blocks=[4],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[4, 4],
        stage2_num_channels=[64, 128],
        stage3_num_modules=4,
        stage3_num_blocks=[4, 4, 4],
        stage3_num_channels=[64, 128, 256],
        stage4_num_modules=3,
        stage4_num_blocks=[4, 4, 4, 4],
        stage4_num_channels=[64, 128, 256, 512],
        **kwargs)
    return model

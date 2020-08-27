# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import os
import math

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, Dropout
from paddle.fluid.dygraph import SyncBatchNorm as BatchNorm

from dygraph.utils import utils

from dygraph.cvlibs import manager

__all__ = [
    "ResNet18_vd", "ResNet34_vd", "ResNet50_vd", "ResNet101_vd", "ResNet152_vd"
]


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(
            self,
            num_channels,
            num_filters,
            filter_size,
            stride=1,
            dilation=1,
            groups=1,
            is_vd_mode=False,
            act=None,
            name=None, ):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = Pool2D(
            pool_size=2, pool_stride=2, pool_padding=0, pool_type='avg', ceil_mode=True)
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2 if dilation ==1  else 0,
            dilation=dilation,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 if_first=False,
                 dilation=1,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a")

        self.dilation = dilation

        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            dilation=dilation,
            name=name + "_branch2b")
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c")

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=1,
                is_vd_mode=False if if_first or stride==1 else True,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        
        ####################################################################
        # If given dilation rate > 1, using corresponding padding
        if self.dilation > 1:
            padding = self.dilation
            y = fluid.layers.pad(y, [0,0,0,0,padding,padding,padding,padding])
        #####################################################################
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv2)
        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(y)


class BasicBlock(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            name=name + "_branch2b")

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = fluid.layers.elementwise_add(x=short, y=conv1)
        
        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(y)


class ResNet_vd(fluid.dygraph.Layer):
    def __init__(self, layers=50, class_dim=1000, output_stride=None, multi_grid=(1, 2, 4), **kwargs):
        super(ResNet_vd, self).__init__()
        
        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512,
                        1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        dilation_dict=None
        if output_stride == 8:
            dilation_dict = {2: 2, 3: 4}
        elif output_stride == 16:
            dilation_dict = {3: 2}

        self.conv1_1 = ConvBNLayer(
            num_channels=3,
            num_filters=32,
            filter_size=3,
            stride=2,
            act='relu',
            name="conv1_1")
        self.conv1_2 = ConvBNLayer(
            num_channels=32,
            num_filters=32,
            filter_size=3,
            stride=1,
            act='relu',
            name="conv1_2")
        self.conv1_3 = ConvBNLayer(
            num_channels=32,
            num_filters=64,
            filter_size=3,
            stride=1,
            act='relu',
            name="conv1_3")
        self.pool2d_max = Pool2D(
            pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        
        # self.block_list = []
        self.stage_list = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                block_list=[]
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    
                    ###############################################################################
                    # Add dilation rate for some segmentation tasks, if dilation_dict is not None.
                    dilation_rate = dilation_dict[block] if dilation_dict and block in dilation_dict else 1
                    
                    # Actually block here is 'stage', and i is 'block' in 'stage'
                    # At the stage 4, expand the the dilation_rate using multi_grid, default (1, 2, 4)
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]
                    #print("stage {}, block {}: dilation rate".format(block, i), dilation_rate)
                    ###############################################################################

                    bottleneck_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BottleneckBlock(
                            num_channels=num_channels[block] if i == 0 else num_filters[block] * 4,
                            num_filters=num_filters[block],
                            stride=2 if i == 0 and block != 0 and dilation_rate == 1 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            name=conv_name,
                            dilation=dilation_rate))

                    block_list.append(bottleneck_block)
                    shortcut = True
                self.stage_list.append(block_list)
        else:
            for block in range(len(depth)):
                shortcut = False
                block_list=[]
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BasicBlock(
                            num_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            num_filters=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            name=conv_name))
                    block_list.append(basic_block)
                    shortcut = True
                self.stage_list.append(block_list)

        self.pool2d_avg = Pool2D(
            pool_size=7, pool_type='avg', global_pooling=True)

        self.pool2d_avg_channels = num_channels[-1] * 2

        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)

        self.out = Linear(
            self.pool2d_avg_channels,
            class_dim,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name="fc_0.w_0"),
            bias_attr=ParamAttr(name="fc_0.b_0"))

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)

        # A feature list saves the output feature map of each stage.
        feat_list = []
        for i, stage in enumerate(self.stage_list):
            for j, block in enumerate(stage):
                y = block(y)
                #print("stage {} block {}".format(i+1, j+1), y.shape)
            feat_list.append(y)
            
        y = self.pool2d_avg(y)
        y = fluid.layers.reshape(y, shape=[-1, self.pool2d_avg_channels])
        y = self.out(y)
        return y, feat_list

    # def init_weight(self, pretrained_model=None):

    #     if pretrained_model is not None:
    #         if os.path.exists(pretrained_model):
    #             utils.load_pretrained_model(self, pretrained_model)

            



def ResNet18_vd(**args):
    model = ResNet_vd(layers=18, **args)
    return model


def ResNet34_vd(**args):
    model = ResNet_vd(layers=34, **args)
    return model

@manager.BACKBONES.add_component
def ResNet50_vd(**args):
    model = ResNet_vd(layers=50, **args)
    return model

@manager.BACKBONES.add_component
def ResNet101_vd(**args):
    model = ResNet_vd(layers=101, **args)
    return model


def ResNet152_vd(**args):
    model = ResNet_vd(layers=152, **args)
    return model


def ResNet200_vd(**args):
    model = ResNet_vd(layers=200, **args)
    return model
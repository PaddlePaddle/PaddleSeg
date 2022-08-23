# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

# reference: https://arxiv.org/pdf/1512.03385

from __future__ import absolute_import, division, print_function
import os,sys
sys.path.insert(0,os.getcwd()+'//..//..')
import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm, Linear, BatchNorm2D
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform
from paddle.regularizer import L2Decay
import math
from medicalseg.cvlibs import manager
from medicalseg.utils import utils




MODEL_URLS = {
    "ResNet18":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet18_pretrained.pdparams",
    "ResNet18_vd":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet18_vd_pretrained.pdparams",
    "ResNet34":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet34_pretrained.pdparams",
    "ResNet34_vd":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet34_vd_pretrained.pdparams",
    "ResNet50":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_pretrained.pdparams",
    "ResNet50_vd":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_pretrained.pdparams",
    "ResNet101":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet101_pretrained.pdparams",
    "ResNet101_vd":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet101_vd_pretrained.pdparams",
    "ResNet152":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet152_pretrained.pdparams",
    "ResNet152_vd":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet152_vd_pretrained.pdparams",
    "ResNet200_vd":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet200_vd_pretrained.pdparams",
}

MODEL_STAGES_PATTERN = {
    "ResNet18": ["blocks[1]", "blocks[3]", "blocks[5]", "blocks[7]"],
    "ResNet34": ["blocks[2]", "blocks[6]", "blocks[12]", "blocks[15]"],
    "ResNet50": ["blocks[2]", "blocks[6]", "blocks[12]", "blocks[15]"],
    "ResNet101": ["blocks[2]", "blocks[6]", "blocks[29]", "blocks[32]"],
    "ResNet152": ["blocks[2]", "blocks[10]", "blocks[46]", "blocks[49]"],
    "ResNet200": ["blocks[2]", "blocks[14]", "blocks[62]", "blocks[65]"]
}

__all__ = MODEL_URLS.keys()
'''
ResNet config: dict.
    key: depth of ResNet.
    values: config's dict of specific model.
        keys:
            block_type: Two different blocks in ResNet, BasicBlock and BottleneckBlock are optional.
            block_depth: The number of blocks in different stages in ResNet.
            num_channels: The number of channels to enter the next stage.
'''
NET_CONFIG = {
    "18": {
        "block_type": "BasicBlock",
        "block_depth": [2, 2, 2, 2],
        "num_channels": [64, 64, 128, 256]
    },
    "34": {
        "block_type": "BasicBlock",
        "block_depth": [3, 4, 6, 3],
        "num_channels": [64, 64, 128, 256]
    },
    "50": {
        "block_type": "BottleneckBlock",
        "block_depth": [3, 4, 6, 3],
        "num_channels": [64, 256, 512, 1024]
    },
    "101": {
        "block_type": "BottleneckBlock",
        "block_depth": [3, 4, 23, 3],
        "num_channels": [64, 256, 512, 1024]
    },
    "152": {
        "block_type": "BottleneckBlock",
        "block_depth": [3, 8, 36, 3],
        "num_channels": [64, 256, 512, 1024]
    },
    "200": {
        "block_type": "BottleneckBlock",
        "block_depth": [3, 12, 48, 3],
        "num_channels": [64, 256, 512, 1024]
    },
}


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 is_vd_mode=False,
                 act=None,
                 lr_mult=1.0,
                 data_format="NCHW"):
        super().__init__()
        self.is_vd_mode = is_vd_mode
        self.act = act
        self.avg_pool = AvgPool2D(
            kernel_size=2, stride=stride, padding="SAME", ceil_mode=True)
        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=1 if is_vd_mode else stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=False,
            data_format=data_format)

        self.bn = BatchNorm(
            num_filters,
            param_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult),
            data_layout=data_format)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.is_vd_mode:
            x = self.avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 if_first=False,
                 lr_mult=1.0,
                 data_format="NCHW"):
        super().__init__()
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act="relu",
            lr_mult=lr_mult,
            data_format=data_format)
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="relu",
            lr_mult=lr_mult,
            data_format=data_format)
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            lr_mult=lr_mult,
            data_format=data_format)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride,
                is_vd_mode=False if if_first else True,
                lr_mult=lr_mult,
                data_format=data_format)

        self.relu = nn.ReLU()
        self.shortcut = shortcut

    def forward(self, x):
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        if self.shortcut:
            short = identity
        else:
            short = self.short(identity)
        x = paddle.add(x=x, y=short)
        x = self.relu(x)
        return x


class BasicBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 if_first=False,
                 lr_mult=1.0,
                 data_format="NCHW"):
        super().__init__()

        self.stride = stride
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="relu",
            lr_mult=lr_mult,
            data_format=data_format)
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            lr_mult=lr_mult,
            data_format=data_format)
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=1,
                stride=stride,
                is_vd_mode=False if if_first else True,
                lr_mult=lr_mult,
                data_format=data_format)
        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        if self.shortcut:
            short = identity
        else:
            short = self.short(identity)
        x = paddle.add(x=x, y=short)
        x = self.relu(x)
        return x


class ResNet(nn.Layer):
    """
    ResNet
    Args:
        config: dict. config of ResNet.
        version: str="vb". Different version of ResNet, version vd can perform better. 
        class_num: int=1000. The number of classes.
        lr_mult_list: list. Control the learning rate of different stages.
    Returns:
        model: nn.Layer. Specific ResNet model depends on args.
    """

    def __init__(self,
                 config,
                 stages_pattern,
                 version="vb",
                 stem_act="relu",
                 class_num=1000,
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0],
                 stride_list=[2, 2, 2, 2, 2],
                 data_format="NCHW",
                 input_image_channel=3,
                 return_patterns=None,
                 return_stages=None,
                 **kargs):
        super().__init__()

        self.cfg = config
        self.lr_mult_list = lr_mult_list
        self.stride_list = stride_list
        self.is_vd_mode = version == "vd"
        self.class_num = class_num
        self.num_filters = [64, 128, 256, 512]
        self.block_depth = self.cfg["block_depth"]
        self.block_type = self.cfg["block_type"]
        self.num_channels = self.cfg["num_channels"]
        self.channels_mult = 1 if self.num_channels[-1] == 256 else 4

        assert isinstance(self.lr_mult_list, (
            list, tuple
        )), "lr_mult_list should be in (list, tuple) but got {}".format(
            type(self.lr_mult_list))
        if len(self.lr_mult_list) != 5:
            msg = "lr_mult_list length should be 5 but got {}, default lr_mult_list used".format(
                len(self.lr_mult_list))
            self.lr_mult_list = [1.0, 1.0, 1.0, 1.0, 1.0]

        assert isinstance(self.stride_list, (
            list, tuple
        )), "stride_list should be in (list, tuple) but got {}".format(
            type(self.stride_list))
        assert len(self.stride_list
                   ) == 5, "stride_list length should be 5 but got {}".format(
                       len(self.stride_list))

        self.stem_cfg = {
            #num_channels, num_filters, filter_size, stride
            "vb": [[input_image_channel, 64, 7, self.stride_list[0]]],
            "vd": [[input_image_channel, 32, 3, self.stride_list[0]],
                   [32, 32, 3, 1], [32, 64, 3, 1]]
        }

        self.stem = nn.Sequential(*[
            ConvBNLayer(
                num_channels=in_c,
                num_filters=out_c,
                filter_size=k,
                stride=s,
                act=stem_act,
                lr_mult=self.lr_mult_list[0],
                data_format=data_format)
            for in_c, out_c, k, s in self.stem_cfg[version]
        ])

        self.max_pool = MaxPool2D(
            kernel_size=3,
            stride=stride_list[1],
            padding=1,
            data_format=data_format)
        block_list = []
        for block_idx in range(len(self.block_depth)):
            # small_block_list=[]
            shortcut = False
            for i in range(self.block_depth[block_idx]):
                # small_block_list.append(globals()[self.block_type](
                #     num_channels=self.num_channels[block_idx] if i == 0 else
                #     self.num_filters[block_idx] * self.channels_mult,
                #     num_filters=self.num_filters[block_idx],
                #     stride=self.stride_list[block_idx + 1]
                #     if i == 0 and block_idx != 0 else 1,
                #     shortcut=shortcut,
                #     if_first=block_idx == i == 0 if version == "vd" else True,
                #     lr_mult=self.lr_mult_list[block_idx + 1],
                #     data_format=data_format))
                block_list.append(globals()[self.block_type](
                    num_channels=self.num_channels[block_idx] if i == 0 else
                    self.num_filters[block_idx] * self.channels_mult,
                    num_filters=self.num_filters[block_idx],
                    stride=self.stride_list[block_idx + 1]
                    if i == 0 and block_idx != 0 else 1,
                    shortcut=shortcut,
                    if_first=block_idx == i == 0 if version == "vd" else True,
                    lr_mult=self.lr_mult_list[block_idx + 1],
                    data_format=data_format))
                shortcut = True
            # block_list.append(nn.Sequential(*small_block_list))
        self.blocks = nn.LayerList(block_list)

        self.avg_pool = AdaptiveAvgPool2D(1, data_format=data_format)
        self.flatten = nn.Flatten()
        self.avg_pool_channels = self.num_channels[-1] * 2
        stdv = 1.0 / math.sqrt(self.avg_pool_channels * 1.0)
        self.fc = Linear(
            self.avg_pool_channels,
            self.class_num,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))

        self.data_format = data_format
        self.features=[]



    def forward(self, x):
        features=[]
        with paddle.static.amp.fp16_guard():
            if self.data_format == "NHWC":
                x = paddle.transpose(x, [0, 2, 3, 1])
                x.stop_gradient = True
            x = self.stem(x)
            features.append(x)
            x = self.max_pool(x)
            tag_list=[]
            count=0
            for depth in self.block_depth:
                count=count+depth
                tag_list.append(count-1)


            for index,block in enumerate(self.blocks):
                x=block(x)
                if index in tag_list:
                    features.append(x)

            # x = self.blocks(x)
            # x = self.avg_pool(x)
            # x = self.flatten(x)
            # x = self.fc(x)
        return features[3],features[2::-1]


    def init_weight(self,pretrained,url):
        if pretrained:
            utils.load_entire_model(self, url)

def ResNet50(pretrained=False, use_ssld=False, **kwargs):
    """
    ResNet50
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ResNet50` model depends on args.
    """
    model = ResNet(
        config=NET_CONFIG["50"],
        stages_pattern=MODEL_STAGES_PATTERN["ResNet50"],
        version="vb",
        **kwargs)
    model.init_weight(pretrained,MODEL_URLS["ResNet50"])
    return model



if __name__=="__main__":
    model=ResNet50()
    test_x=paddle.rand([1,3,224,224])
    x,feature_list=model(test_x)
    print(x.shape)
    for feature in feature_list:
        print(feature.shape)


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

import warnings

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils
from paddleseg.models.backbones.top_transformer import ConvBNAct


@manager.MODELS.add_component
class TopFormer(nn.Layer):
    """
    The Token Pyramid Transformer(TopFormer) implementation based on PaddlePaddle.

    The original article refers to
    Zhang, Wenqiang, Zilong Huang, Guozhong Luo, Tao Chen, Xinggang Wang, Wenyu Liu, Gang Yu,
    and Chunhua Shen. "TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation." 
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    pp. 12083-12093. 2022.

    This model refers to https://github.com/hustvl/TopFormer.

    Args:
        num_classes(int,optional): The unique number of target classes.
        backbone(nn.Layer): Backbone network.
        head_use_dw (bool, optional): Whether the head use depthwise convolutions. Default: False.
        align_corners (bool, optional): Set the align_corners in resizing. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 head_use_dw=False,
                 align_corners=False,
                 pretrained=None):
        super().__init__()
        self.backbone = backbone

        head_in_channels = [
            i for i in backbone.injection_out_channels if i is not None
        ]
        self.decode_head = TopFormerHead(num_classes=num_classes,
                                         in_channels=head_in_channels,
                                         use_dw=head_use_dw,
                                         align_corners=align_corners)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        x_hw = x.shape[2:]
        x = self.backbone(x)  # len=3, 1/8,1/16,1/32
        x = self.decode_head(x)
        x = F.interpolate(x,
                          x_hw,
                          mode='bilinear',
                          align_corners=self.align_corners)

        return [x]


class TopFormerHead(nn.Layer):

    def __init__(self,
                 num_classes,
                 in_channels,
                 in_index=[0, 1, 2],
                 in_transform='multiple_select',
                 use_dw=False,
                 dropout_ratio=0.1,
                 align_corners=False):
        super().__init__()

        self.in_index = in_index
        self.in_transform = in_transform
        self.align_corners = align_corners

        self._init_inputs(in_channels, in_index, in_transform)
        self.linear_fuse = ConvBNAct(in_channels=self.last_channels,
                                     out_channels=self.last_channels,
                                     kernel_size=1,
                                     stride=1,
                                     groups=self.last_channels if use_dw else 1,
                                     act=nn.ReLU)
        self.dropout = nn.Dropout2D(dropout_ratio)
        self.conv_seg = nn.Conv2D(self.last_channels,
                                  num_classes,
                                  kernel_size=1)

    def _init_inputs(self, in_channels, in_index, in_transform):
        assert in_transform in [None, 'resize_concat', 'multiple_select']
        if in_transform is not None:
            assert len(in_channels) == len(in_index)
            if in_transform == 'resize_concat':
                self.last_channels = sum(in_channels)
            else:
                self.last_channels = in_channels[0]
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.last_channels = in_channels

    def _transform_inputs(self, inputs):
        if self.in_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            inputs = [
                F.interpolate(input_data=x,
                              size=inputs[0].shape[2:],
                              mode='bilinear',
                              align_corners=self.align_corners) for x in inputs
            ]
            inputs = paddle.concat(inputs, axis=1)
        elif self.in_transform == 'multiple_select':
            inputs_tmp = [inputs[i] for i in self.in_index]
            inputs = inputs_tmp[0]
            for x in inputs_tmp[1:]:
                x = F.interpolate(x,
                                  size=inputs.shape[2:],
                                  mode='bilinear',
                                  align_corners=self.align_corners)
                inputs += x
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, x):
        x = self._transform_inputs(x)
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        return x

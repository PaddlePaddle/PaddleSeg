# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddleseg.models.backbones.strideformer import ConvBNAct


@manager.MODELS.add_component
class PPMobileSeg(nn.Layer):
    """
    The PP_MobileSeg implementation based on PaddlePaddle.

    The original article refers to "Shiyu Tang, Ting Sun, Juncai Peng, Guowei Chen, Yuying Hao, 
    Manhui Lin, Zhihong Xiao, Jiangbin You, Yi Liu. PP-MobileSeg: Explore the Fast and Accurate 
    Semantic Segmentation Model on Mobile Devices. https://arxiv.org/abs/2304.05152"


    Args:
        num_classes(int): The unique number of target classes.
        backbone(nn.Layer): Backbone network.
        head_use_dw (bool, optional): Whether the head use depthwise convolutions. Default: True.
        align_corners (bool, optional): Set the align_corners in resizing. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
        upsample (str, optional): The type of upsample module, valid for VIM is recommend to be used during inference. Default: intepolate.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 head_use_dw=True,
                 align_corners=False,
                 pretrained=None,
                 upsample='intepolate'):
        super().__init__()
        self.backbone = backbone
        self.upsample = upsample
        self.num_classes = num_classes

        self.decode_head = PPMobileSegHead(
            num_classes=num_classes,
            in_channels=backbone.feat_channels[0],
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
        x = self.backbone(x)
        x = self.decode_head(x)
        if self.upsample == 'intepolate' or self.training or self.num_classes < 30:
            x = F.interpolate(
                x, x_hw, mode='bilinear', align_corners=self.align_corners)
        elif self.upsample == 'vim':
            labelset = paddle.unique(paddle.argmax(x, 1))
            x = paddle.gather(x, labelset, axis=1)
            x = F.interpolate(
                x, x_hw, mode='bilinear', align_corners=self.align_corners)

            pred = paddle.argmax(x, 1)
            pred_retrieve = paddle.zeros(pred.shape, dtype='int32')
            for i, val in enumerate(labelset):
                pred_retrieve[pred == i] = labelset[i].cast('int32')

            x = pred_retrieve
        else:
            raise NotImplementedError(self.upsample, " is not implemented")

        return [x]


class PPMobileSegHead(nn.Layer):
    def __init__(self,
                 num_classes,
                 in_channels,
                 use_dw=False,
                 dropout_ratio=0.1,
                 align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.last_channels = in_channels

        self.linear_fuse = ConvBNAct(
            in_channels=self.last_channels,
            out_channels=self.last_channels,
            kernel_size=1,
            stride=1,
            groups=self.last_channels if use_dw else 1,
            act=nn.ReLU)
        self.dropout = nn.Dropout2D(dropout_ratio)
        self.conv_seg = nn.Conv2D(
            self.last_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        return x

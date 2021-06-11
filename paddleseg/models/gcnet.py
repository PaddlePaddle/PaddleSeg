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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class GCNet(nn.Layer):
    """
    The GCNet implementation based on PaddlePaddle.

    The original article refers to
    Cao, Yue, et al. "GCnet: Non-local networks meet squeeze-excitation networks and beyond"
    (https://arxiv.org/pdf/1904.11492.pdf).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network, currently support Resnet50/101.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
        gc_channels (int, optional): The input channels to Global Context Block. Default: 512.
        ratio (float, optional): It indicates the ratio of attention channels and gc_channels. Default: 0.25.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(2, 3),
                 gc_channels=512,
                 ratio=0.25,
                 enable_auxiliary_loss=True,
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = GCNetHead(num_classes, backbone_indices, backbone_channels,
                              gc_channels, ratio, enable_auxiliary_loss)
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        return [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class GCNetHead(nn.Layer):
    """
    The GCNetHead implementation.

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): Two values in the tuple indicate the indices of output of backbone.
            The first index will be taken as a deep-supervision feature in auxiliary layer;
            the second one will be taken as input of GlobalContextBlock.
        backbone_channels (tuple): The same length with "backbone_indices". It indicates the channels of corresponding index.
        gc_channels (int): The input channels to Global Context Block.
        ratio (float): It indicates the ratio of attention channels and gc_channels.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
    """

    def __init__(self,
                 num_classes,
                 backbone_indices,
                 backbone_channels,
                 gc_channels,
                 ratio,
                 enable_auxiliary_loss=True):

        super().__init__()

        in_channels = backbone_channels[1]
        self.conv_bn_relu1 = layers.ConvBNReLU(
            in_channels=in_channels,
            out_channels=gc_channels,
            kernel_size=3,
            padding=1)

        self.gc_block = GlobalContextBlock(
            gc_channels=gc_channels, in_channels=gc_channels, ratio=ratio)

        self.conv_bn_relu2 = layers.ConvBNReLU(
            in_channels=gc_channels,
            out_channels=gc_channels,
            kernel_size=3,
            padding=1)

        self.conv_bn_relu3 = layers.ConvBNReLU(
            in_channels=in_channels + gc_channels,
            out_channels=gc_channels,
            kernel_size=3,
            padding=1)

        self.dropout = nn.Dropout(p=0.1)

        self.conv = nn.Conv2D(
            in_channels=gc_channels, out_channels=num_classes, kernel_size=1)

        if enable_auxiliary_loss:
            self.auxlayer = layers.AuxLayer(
                in_channels=backbone_channels[0],
                inter_channels=backbone_channels[0] // 4,
                out_channels=num_classes)

        self.backbone_indices = backbone_indices
        self.enable_auxiliary_loss = enable_auxiliary_loss

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[1]]

        output = self.conv_bn_relu1(x)
        output = self.gc_block(output)
        output = self.conv_bn_relu2(output)

        output = paddle.concat([x, output], axis=1)
        output = self.conv_bn_relu3(output)

        output = self.dropout(output)
        logit = self.conv(output)
        logit_list.append(logit)

        if self.enable_auxiliary_loss:
            low_level_feat = feat_list[self.backbone_indices[0]]
            auxiliary_logit = self.auxlayer(low_level_feat)
            logit_list.append(auxiliary_logit)

        return logit_list


class GlobalContextBlock(nn.Layer):
    """
    Global Context Block implementation.

    Args:
        in_channels (int): The input channels of Global Context Block.
        ratio (float): The channels of attention map.
    """

    def __init__(self, gc_channels, in_channels, ratio):
        super().__init__()
        self.gc_channels = gc_channels

        self.conv_mask = nn.Conv2D(
            in_channels=in_channels, out_channels=1, kernel_size=1)

        self.softmax = nn.Softmax(axis=2)

        inter_channels = int(in_channels * ratio)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2D(
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=1),
            nn.LayerNorm(normalized_shape=[inter_channels, 1, 1]), nn.ReLU(),
            nn.Conv2D(
                in_channels=inter_channels,
                out_channels=in_channels,
                kernel_size=1))

    def global_context_block(self, x):
        x_shape = paddle.shape(x)

        # [N, C, H * W]
        input_x = paddle.reshape(x, shape=[0, self.gc_channels, -1])
        # [N, 1, C, H * W]
        input_x = paddle.unsqueeze(input_x, axis=1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = paddle.reshape(context_mask, shape=[0, 1, -1])
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = paddle.unsqueeze(context_mask, axis=-1)
        # [N, 1, C, 1]
        context = paddle.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = paddle.reshape(context, shape=[0, self.gc_channels, 1, 1])

        return context

    def forward(self, x):
        context = self.global_context_block(x)
        channel_add_term = self.channel_add_conv(context)
        out = x + channel_add_term
        return out

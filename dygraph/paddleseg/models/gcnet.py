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

import os

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddleseg.cvlibs import manager
from paddleseg.models.common.layer_libs import ConvBNReLU, AuxLayer
from paddleseg.utils import utils




@manager.MODELS.add_component
class GCNet(nn.Layer):
    """
    The GCNet implementation based on PaddlePaddle.

    The original article refers to 
        Cao, Yue, et al. "GCnet: Non-local networks meet squeeze-excitation networks and beyond."
        (https://arxiv.org/pdf/1904.11492.pdf)

    Args:
        num_classes (int): the unique number of target classes.
        backbone (Paddle.nn.Layer): backbone network, currently support Resnet50/101.
        backbone_indices (tuple): two values in the tuple indicate the indices of output of backbone.
        gc_channels (int): input channels to Global Context Block. Default to 512.
        ratio (float): it indicates the ratio of attention channels and gc_channels. Default to 1/4.
        enable_auxiliary_loss (bool): a bool values indicates whether adding auxiliary loss. Default to True.
        pretrained (str): the path of pretrained model. Default to None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(2, 3),
                 gc_channels=512,
                 ratio=1 / 4,
                 enable_auxiliary_loss=True,
                 pretrained=None):

        super(GCNet, self).__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = GCNetHead(
            num_classes, 
            backbone_indices,
            backbone_channels,
            gc_channels,
            ratio,
            enable_auxiliary_loss)

        utils.load_entire_model(self, pretrained)

    def forward(self, input):
        feat_list = self.backbone(input)
        logit_list = self.head(feat_list)
        return [
            F.resize_bilinear(logit, input.shape[2:]) for logit in logit_list
        ]


class GCNetHead(nn.Layer):
    """
    The GCNetHead implementation.

    Args:
        num_classes (int): the unique number of target classes.
        backbone_indices (tuple): two values in the tuple indicate the indices of output of backbone.
            the first index will be taken as a deep-supervision feature in auxiliary layer;
            the second one will be taken as input of GlobalContextBlock. Usually backbone 
            consists of four downsampling stage, and return an output of each stage, so we 
            set default (2, 3), which means taking feature map of the third stage (res4b22) 
            and the fourth stage (res5c) in backbone.
        backbone_channels (tuple): the same length with "backbone_indices". It indicates the channels of corresponding index.
        gc_channels (int): input channels to Global Context Block. Default to 512.
        ratio (float): it indicates the ratio of attention channels and gc_channels. Default to 1/4.
        enable_auxiliary_loss (bool): a bool values indicates whether adding auxiliary loss. Default to True.
    """

    def __init__(self,
                 num_classes,
                 backbone_indices=(2, 3),
                 backbone_channels=(1024, 2048),
                 gc_channels=512,
                 ratio=1 / 4,
                 enable_auxiliary_loss=True):

        super(GCNetHead, self).__init__()

        in_channels = backbone_channels[1]
        self.conv_bn_relu1 = ConvBNReLU(
            in_channels=in_channels,
            out_channels=gc_channels,
            kernel_size=3,
            padding=1)

        self.gc_block = GlobalContextBlock(in_channels=gc_channels, ratio=ratio)

        self.conv_bn_relu2 = ConvBNReLU(
            in_channels=gc_channels,
            out_channels=gc_channels,
            kernel_size=3,
            padding=1)

        self.conv_bn_relu3 = ConvBNReLU(
            in_channels=in_channels + gc_channels,
            out_channels=gc_channels,
            kernel_size=3,
            padding=1)

        self.conv = nn.Conv2d(
            in_channels=gc_channels, out_channels=num_classes, kernel_size=1)

        if enable_auxiliary_loss:
            self.auxlayer = AuxLayer(
                in_channels=backbone_channels[0],
                inter_channels=backbone_channels[0] // 4,
                out_channels=num_classes)

        self.backbone_indices = backbone_indices
        self.enable_auxiliary_loss = enable_auxiliary_loss

        self.init_weight()

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[1]]

        output = self.conv_bn_relu1(x)
        output = self.gc_block(output)
        output = self.conv_bn_relu2(output)

        output = paddle.concat([x, output], axis=1)
        output = self.conv_bn_relu3(output)

        output = F.dropout(output, p=0.1)  # dropout_prob
        logit = self.conv(output)
        logit_list.append(logit)

        if self.enable_auxiliary_loss:
            low_level_feat = feat_list[self.backbone_indices[0]]
            auxiliary_logit = self.auxlayer(low_level_feat)
            logit_list.append(auxiliary_logit)

        return logit_list

    def init_weight(self, pretrained_model=None):
        """
        Initialize the parameters of model parts.
        """
        pass


class GlobalContextBlock(nn.Layer):
    """
    Global Context Block implementation.

    Args:
        in_channels (int): input channels of Global Context Block
        ratio (float): the channels of attention map.
    """

    def __init__(self, in_channels, ratio):
        super(GlobalContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=1)

        self.softmax = nn.Softmax(axis=2)

        inter_channels = int(in_channels * ratio)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=1),
            nn.LayerNorm(normalized_shape=[inter_channels, 1, 1]), nn.ReLU(),
            nn.Conv2d(
                in_channels=inter_channels,
                out_channels=in_channels,
                kernel_size=1))

    def global_context_block(self, x):
        batch, channel, height, width = x.shape

        # [N, C, H * W]
        input_x = paddle.reshape(x, shape=[batch, channel, height * width])
        # [N, 1, C, H * W]
        input_x = paddle.unsqueeze(input_x, axis=1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = paddle.reshape(
            context_mask, shape=[batch, 1, height * width])
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = paddle.unsqueeze(context_mask, axis=-1)
        # [N, 1, C, 1]
        context = paddle.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = paddle.reshape(context, shape=[batch, channel, 1, 1])

        return context

    def forward(self, x):
        context = self.global_context_block(x)
        channel_add_term = self.channel_add_conv(context)
        out = x + channel_add_term
        return out

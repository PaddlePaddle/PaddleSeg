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
from paddleseg.models.common import pyramid_pool
from paddleseg.models.common.layer_libs import ConvBNReLU, SeparableConvBNReLU, AuxLayer
from paddleseg.utils import utils

__all__ = ['DeepLabV3P', 'DeepLabV3']


@manager.MODELS.add_component
class DeepLabV3P(nn.Layer):
    """
    The DeepLabV3Plus implementation based on PaddlePaddle.

    The original article refers to
     Liang-Chieh Chen, et, al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
     (https://arxiv.org/abs/1802.02611)

    Args:
        num_classes (int): the unique number of target classes.
        backbone (paddle.nn.Layer): backbone network, currently support Resnet50_vd/Resnet101_vd/Xception65.
        backbone_indices (tuple): two values in the tuple indicate the indices of output of backbone.
            the first index will be taken as a low-level feature in Decoder component;
            the second one will be taken as input of ASPP component.
            Usually backbone consists of four downsampling stage, and return an output of
            each stage, so we set default (0, 3), which means taking feature map of the first
            stage in backbone as low-level feature used in Decoder, and feature map of the fourth
            stage as input of ASPP.
        aspp_ratios (tuple): the dilation rate using in ASSP module.
            if output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            if output_stride=8, aspp_ratios is (1, 12, 24, 36).
        aspp_out_channels (int): the output channels of ASPP module.
        pretrained (str): the path of pretrained model. Default to None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(0, 3),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256,
                 pretrained=None):

        super(DeepLabV3P, self).__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = DeepLabV3PHead(num_classes, backbone_indices,
                                   backbone_channels, aspp_ratios,
                                   aspp_out_channels)

        utils.load_entire_model(self, pretrained)

    def forward(self, input):
        feat_list = self.backbone(input)
        logit_list = self.head(feat_list)
        return [
            F.resize_bilinear(logit, input.shape[2:]) for logit in logit_list
        ]


class DeepLabV3PHead(nn.Layer):
    """
    The DeepLabV3PHead implementation based on PaddlePaddle.

    Args:
        num_classes (int): the unique number of target classes.
        backbone_indices (tuple): two values in the tuple indicate the indices of output of backbone.
            the first index will be taken as a low-level feature in Decoder component;
            the second one will be taken as input of ASPP component.
            Usually backbone consists of four downsampling stage, and return an output of
            each stage, so we set default (0, 3), which means taking feature map of the first
            stage in backbone as low-level feature used in Decoder, and feature map of the fourth
            stage as input of ASPP.
        backbone_channels (tuple): the same length with "backbone_indices". It indicates the channels of corresponding index.
        aspp_ratios (tuple): the dilation rate using in ASSP module.
            if output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            if output_stride=8, aspp_ratios is (1, 12, 24, 36).
        aspp_out_channels (int): the output channels of ASPP module.

    """

    def __init__(self,
                 num_classes,
                 backbone_indices,
                 backbone_channels,
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256):

        super(DeepLabV3PHead, self).__init__()

        self.aspp = pyramid_pool.ASPPModule(
            aspp_ratios,
            backbone_channels[1],
            aspp_out_channels,
            sep_conv=True,
            image_pooling=True)
        self.decoder = Decoder(num_classes, backbone_channels[0])
        self.backbone_indices = backbone_indices
        self.init_weight()

    def forward(self, feat_list):
        logit_list = []
        low_level_feat = feat_list[self.backbone_indices[0]]
        x = feat_list[self.backbone_indices[1]]
        x = self.aspp(x)
        logit = self.decoder(x, low_level_feat)
        logit_list.append(logit)

        return logit_list

    def init_weight(self):
        pass


@manager.MODELS.add_component
class DeepLabV3(nn.Layer):
    """
    The DeepLabV3 implementation based on PaddlePaddle.

    The original article refers to
     Liang-Chieh Chen, et, al. "Rethinking Atrous Convolution for Semantic Image Segmentation"
     (https://arxiv.org/pdf/1706.05587.pdf)

    Args:
        Refer to DeepLabV3P above
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 pretrained=None,
                 backbone_indices=(3, ),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256):

        super(DeepLabV3, self).__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = DeepLabV3Head(num_classes, backbone_indices,
                                  backbone_channels, aspp_ratios,
                                  aspp_out_channels)

        utils.load_entire_model(self, pretrained)

    def forward(self, input):
        feat_list = self.backbone(input)
        logit_list = self.head(feat_list)
        return [
            F.resize_bilinear(logit, input.shape[2:]) for logit in logit_list
        ]


class DeepLabV3Head(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone_indices=(3, ),
                 backbone_channels=(2048, ),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256):

        super(DeepLabV3Head, self).__init__()

        self.aspp = pyramid_pool.ASPPModule(
            aspp_ratios,
            backbone_channels[0],
            aspp_out_channels,
            sep_conv=False,
            image_pooling=True)

        self.cls = nn.Conv2d(
            in_channels=aspp_out_channels,
            out_channels=num_classes,
            kernel_size=1)

        self.backbone_indices = backbone_indices
        self.init_weight()

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[0]]
        x = self.aspp(x)
        logit = self.cls(x)
        logit_list.append(logit)

        return logit_list

    def init_weight(self):
        pass


class Decoder(nn.Layer):
    """
    Decoder module of DeepLabV3P model

    Args:
        num_classes (int): the number of classes.
        in_channels (int): the number of input channels in decoder module.

    """

    def __init__(self, num_classes, in_channels):
        super(Decoder, self).__init__()

        self.conv_bn_relu1 = ConvBNReLU(
            in_channels=in_channels, out_channels=48, kernel_size=1)

        self.conv_bn_relu2 = SeparableConvBNReLU(
            in_channels=304, out_channels=256, kernel_size=3, padding=1)
        self.conv_bn_relu3 = SeparableConvBNReLU(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(
            in_channels=256, out_channels=num_classes, kernel_size=1)

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv_bn_relu1(low_level_feat)
        x = F.resize_bilinear(x, low_level_feat.shape[2:])
        x = paddle.concat([x, low_level_feat], axis=1)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.conv(x)
        return x

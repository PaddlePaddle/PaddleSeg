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
from paddleseg.models.common import pyramid_pool, layer_libs
from paddleseg.utils import utils

__all__ = ['DeepLabV3P', 'DeepLabV3']


@manager.MODELS.add_component
class DeepLabV3P(nn.Layer):
    """
    The DeepLabV3Plus implementation based on PaddlePaddle.

    The orginal artile refers to
    "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
     Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.
     (https://arxiv.org/abs/1802.02611)

     The DeepLabV3P consists of three main components, Backbone, ASPP and Decoder.

    Args:
        num_classes (int): the unique number of target classes.
        backbone (paddle.nn.Layer): backbone network, currently support Xception65, Resnet101_vd.
        model_pretrained (str): the path of pretrained model.
        aspp_ratios (tuple): the dilation rate using in ASSP module.
            if output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            if output_stride=8, aspp_ratios is (1, 12, 24, 36).
        backbone_indices (tuple): two values in the tuple indicte the indices of output of backbone.
            the first index will be taken as a low-level feature in Deconder component;
            the second one will be taken as input of ASPP component.
            Usually backbone consists of four downsampling stage, and return an output of
            each stage, so we set default (0, 3), which means taking feature map of the first
            stage in backbone as low-level feature used in Decoder, and feature map of the fourth
            stage as input of ASPP.
        backbone_channels (tuple): the same length with "backbone_indices". It indicates the channels of corresponding index.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_pretrained=None,
                 model_pretrained=None,
                 backbone_indices=(0, 3),
                 backbone_channels=(256, 2048),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256):

        super(DeepLabV3P, self).__init__()

        self.backbone = backbone
        self.backbone_pretrained = backbone_pretrained
        self.model_pretrained = model_pretrained
        
        self.aspp = pyramid_pool.ASPPModule(
            aspp_ratios, backbone_channels[1], aspp_out_channels, sep_conv=True, image_pooling=True)
        self.decoder = Decoder(num_classes, backbone_channels[0])
        self.backbone_indices = backbone_indices
        self.init_weight()

    def forward(self, input, label=None):

        logit_list = []
        _, feat_list = self.backbone(input)
        low_level_feat = feat_list[self.backbone_indices[0]]
        x = feat_list[self.backbone_indices[1]]
        x = self.aspp(x)
        logit = self.decoder(x, low_level_feat)
        logit = F.resize_bilinear(logit, input.shape[2:])
        logit_list.append(logit)

        return logit_list

    def init_weight(self):
        """
        Initialize the parameters of model parts.
        Args:
            pretrained_model ([str], optional): the path of pretrained model. Defaults to None.
        """
        if self.model_pretrained is not None:
            utils.load_pretrained_model(self, self.model_pretrained)
        elif self.backbone_pretrained is not None:
            utils.load_pretrained_model(self.backbone, self.backbone_pretrained)
           

@manager.MODELS.add_component
class DeepLabV3(nn.Layer):
    """
    The DeepLabV3 implementation based on PaddlePaddle.

    The orginal article refers to
    "Rethinking Atrous Convolution for Semantic Image Segmentation"
     Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam.
     (https://arxiv.org/pdf/1706.05587.pdf)

    Args:
        Refer to DeepLabV3P above 
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_pretrained=None,
                 model_pretrained=None,
                 backbone_indices=(3,),
                 backbone_channels=(2048,),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256):

        super(DeepLabV3, self).__init__()

        self.backbone = backbone

        self.aspp = pyramid_pool.ASPPModule(
            aspp_ratios, backbone_channels[0], aspp_out_channels, 
            sep_conv=False, image_pooling=True)

        self.cls = nn.Conv2d(
            in_channels=backbone_channels[0],
            out_channels=num_classes,
            kernel_size=1)

        self.backbone_indices = backbone_indices
        self.init_weight(model_pretrained)

    def forward(self, input, label=None):

        logit_list = []
        _, feat_list = self.backbone(input)
        x = feat_list[self.backbone_indices[0]]
        logit = self.cls(x)
        logit = F.resize_bilinear(logit, input.shape[2:])
        logit_list.append(logit)

        return logit_list

    def init_weight(self, pretrained_model=None):
        """
        Initialize the parameters of model parts.
        Args:
            pretrained_model ([str], optional): the path of pretrained model. Defaults to None.
        """
        if pretrained_model is not None:
            if os.path.exists(pretrained_model):
                utils.load_pretrained_model(self, pretrained_model)
            else:
                raise Exception('Pretrained model is not found: {}'.format(
                    pretrained_model))


class Decoder(nn.Layer):
    """
    Decoder module of DeepLabV3P model

    Args:
        num_classes (int): the number of classes.
        in_channels (int): the number of input channels in decoder module.

    """

    def __init__(self, num_classes, in_channels):
        super(Decoder, self).__init__()

        self.conv_bn_relu1 = layer_libs.ConvBnRelu(
            in_channels=in_channels, out_channels=48, kernel_size=1)

        self.conv_bn_relu2 = layer_libs.DepthwiseConvBnRelu(
            in_channels=304, out_channels=256, kernel_size=3, padding=1)
        self.conv_bn_relu3 = layer_libs.DepthwiseConvBnRelu(
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

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
from paddleseg.models.common import layer_utils
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

        output_stride (int): the ratio of input size and final feature size. 
        Support 16 or 8. Default to 16.

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
                 model_pretrained=None,
                 backbone_indices=(0, 3),
                 backbone_channels=(256, 2048),
                 output_stride=16):

        super(DeepLabV3P, self).__init__()

        self.backbone = backbone
        self.aspp = ASPP(output_stride, backbone_channels[1])
        self.decoder = Decoder(num_classes, backbone_channels[0])
        self.backbone_indices = backbone_indices
        self.init_weight(model_pretrained)

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
                 model_pretrained=None,
                 backbone_indices=(3,),
                 backbone_channels=(2048,),
                 output_stride=16):

        super(DeepLabV3, self).__init__()

        self.backbone = backbone
        self.aspp = ASPP(output_stride, backbone_channels[0])
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


class ImageAverage(nn.Layer):
    """
    Global average pooling

    Args:
        in_channels (int): the number of input channels.

    """

    def __init__(self, in_channels):
        super(ImageAverage, self).__init__()
        self.conv_bn_relu = layer_utils.ConvBnRelu(
            in_channels, out_channels=256, kernel_size=1)

    def forward(self, input):
        x = paddle.reduce_mean(input, dim=[2, 3], keep_dim=True)
        x = self.conv_bn_relu(x)
        x = F.resize_bilinear(x, out_shape=input.shape[2:])
        return x


class ASPP(nn.Layer):
    """
     Decoder module of DeepLabV3P model

    Args:
        output_stride (int): the ratio of input size and final feature size. Support 16 or 8.

        in_channels (int): the number of input channels in decoder module.

    """

    def __init__(self, output_stride, in_channels):
        super(ASPP, self).__init__()

        if output_stride == 16:
            aspp_ratios = (6, 12, 18)
        elif output_stride == 8:
            aspp_ratios = (12, 24, 36)
        else:
            raise NotImplementedError(
                "Only support output_stride is 8 or 16, but received{}".format(
                    output_stride))

        self.image_average = ImageAverage(in_channels=in_channels)

        # The first aspp using 1*1 conv
        self.aspp1 = layer_utils.DepthwiseConvBnRelu(
            in_channels=in_channels, out_channels=256, kernel_size=1)

        # The second aspp using 3*3 (separable) conv at dilated rate aspp_ratios[0]
        self.aspp2 = layer_utils.DepthwiseConvBnRelu(
            in_channels=in_channels,
            out_channels=256,
            kernel_size=3,
            dilation=aspp_ratios[0],
            padding=aspp_ratios[0])

        # The Third aspp using 3*3 (separable) conv at dilated rate aspp_ratios[1]
        self.aspp3 = layer_utils.DepthwiseConvBnRelu(
            in_channels=in_channels,
            out_channels=256,
            kernel_size=3,
            dilation=aspp_ratios[1],
            padding=aspp_ratios[1])

        # The Third aspp using 3*3 (separable) conv at dilated rate aspp_ratios[2]
        self.aspp4 = layer_utils.DepthwiseConvBnRelu(
            in_channels=in_channels,
            out_channels=256,
            kernel_size=3,
            dilation=aspp_ratios[2],
            padding=aspp_ratios[2])

        # After concat op, using 1*1 conv
        self.conv_bn_relu = layer_utils.ConvBnRelu(
            in_channels=1280, out_channels=256, kernel_size=1)

    def forward(self, x):

        x1 = self.image_average(x)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)
        x = paddle.concat([x1, x2, x3, x4, x5], axis=1)

        x = self.conv_bn_relu(x)
        x = F.dropout(x, p=0.1)  # dropout_prob
        return x


class Decoder(nn.Layer):
    """
    Decoder module of DeepLabV3P model

    Args:
        num_classes (int): the number of classes.

        in_channels (int): the number of input channels in decoder module.

    """

    def __init__(self, num_classes, in_channels):
        super(Decoder, self).__init__()

        self.conv_bn_relu1 = layer_utils.ConvBnRelu(
            in_channels=in_channels, out_channels=48, kernel_size=1)

        self.conv_bn_relu2 = layer_utils.DepthwiseConvBnRelu(
            in_channels=304, out_channels=256, kernel_size=3, padding=1)
        self.conv_bn_relu3 = layer_utils.DepthwiseConvBnRelu(
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

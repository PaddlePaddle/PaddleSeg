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
from functools import partial

from paddle import fluid
from paddle.fluid import dygraph
from paddle.fluid.dygraph import Conv2D

from dygraph.utils import utils
from dygraph.models.architectures import layer_utils, xception_deeplab, resnet_vd, mobilenetv3
from dygraph.cvlibs import manager

__all__ = ['DeepLabV3P', "deeplabv3p_resnet101_vd", "deeplabv3p_resnet101_vd_os8",
           "deeplabv3p_resnet50_vd", "deeplabv3p_resnet50_vd_os8",
           "deeplabv3p_xception65_deeplab",
           "deeplabv3p_mobilenetv3_large", "deeplabv3p_mobilenetv3_small"]


class ImageAverage(dygraph.Layer):
    """
    Global average pooling

    Args:
        num_channels (int): the number of input channels.

    """

    def __init__(self, num_channels):
        super(ImageAverage, self).__init__()
        self.conv_bn_relu = layer_utils.ConvBnRelu(num_channels,
                                                   num_filters=256,
                                                   filter_size=1)

    def forward(self, input):
        x = fluid.layers.reduce_mean(input, dim=[2, 3], keep_dim=True)
        x = self.conv_bn_relu(x)
        x = fluid.layers.resize_bilinear(x, out_shape=input.shape[2:])
        return x


class ASPP(dygraph.Layer):
    """
     Decoder module of DeepLabV3P model

    Args:
        output_stride (int): the ratio of input size and final feature size. Support 16 or 8.
        in_channels (int): the number of input channels in decoder module.
        using_sep_conv (bool): whether use separable conv or not. Default to True.
    """

    def __init__(self, output_stride, in_channels, using_sep_conv=True):
        super(ASPP, self).__init__()

        if output_stride == 16:
            aspp_ratios = (6, 12, 18)
        elif output_stride == 8:
            aspp_ratios = (12, 24, 36)
        else:
            raise NotImplementedError("Only support output_stride is 8 or 16, but received{}".format(output_stride))

        self.image_average = ImageAverage(num_channels=in_channels)

        # The first aspp using 1*1 conv
        self.aspp1 = layer_utils.ConvBnRelu(num_channels=in_channels,
                                            num_filters=256,
                                            filter_size=1,
                                            using_sep_conv=False)

        # The second aspp using 3*3 (separable) conv at dilated rate aspp_ratios[0]
        self.aspp2 = layer_utils.ConvBnRelu(num_channels=in_channels,
                                            num_filters=256,
                                            filter_size=3,
                                            using_sep_conv=using_sep_conv,
                                            dilation=aspp_ratios[0],
                                            padding=aspp_ratios[0])

        # The Third aspp using 3*3 (separable) conv at dilated rate aspp_ratios[1]
        self.aspp3 = layer_utils.ConvBnRelu(num_channels=in_channels,
                                            num_filters=256,
                                            filter_size=3,
                                            using_sep_conv=using_sep_conv,
                                            dilation=aspp_ratios[1],
                                            padding=aspp_ratios[1])

        # The Third aspp using 3*3 (separable) conv at dilated rate aspp_ratios[2]
        self.aspp4 = layer_utils.ConvBnRelu(num_channels=in_channels,
                                            num_filters=256,
                                            filter_size=3,
                                            using_sep_conv=using_sep_conv,
                                            dilation=aspp_ratios[2],
                                            padding=aspp_ratios[2])

        # After concat op, using 1*1 conv
        self.conv_bn_relu = layer_utils.ConvBnRelu(num_channels=1280,
                                                   num_filters=256,
                                                   filter_size=1)

    def forward(self, x):

        x1 = self.image_average(x)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)
        x = fluid.layers.concat([x1, x2, x3, x4, x5], axis=1)

        x = self.conv_bn_relu(x)
        x = fluid.layers.dropout(x, dropout_prob=0.1)
        return x


class Decoder(dygraph.Layer):
    """
    Decoder module of DeepLabV3P model

    Args:
        num_classes (int): the number of classes.
        in_channels (int): the number of input channels in decoder module.
        using_sep_conv (bool): whether use separable conv or not. Default to True.

    """

    def __init__(self, num_classes, in_channels, using_sep_conv=True):
        super(Decoder, self).__init__()

        self.conv_bn_relu1 = layer_utils.ConvBnRelu(num_channels=in_channels,
                                                    num_filters=48,
                                                    filter_size=1)

        self.conv_bn_relu2 = layer_utils.ConvBnRelu(num_channels=304,
                                                    num_filters=256,
                                                    filter_size=3,
                                                    using_sep_conv=using_sep_conv,
                                                    padding=1)
        self.conv_bn_relu3 = layer_utils.ConvBnRelu(num_channels=256,
                                                    num_filters=256,
                                                    filter_size=3,
                                                    using_sep_conv=using_sep_conv,
                                                    padding=1)
        self.conv = Conv2D(num_channels=256,
                           num_filters=num_classes,
                           filter_size=1)

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv_bn_relu1(low_level_feat)
        x = fluid.layers.resize_bilinear(x, low_level_feat.shape[2:])
        x = fluid.layers.concat([x, low_level_feat], axis=1)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.conv(x)
        return x


class DeepLabV3P(dygraph.Layer):
    """
    The DeepLabV3P consists of three main components, Backbone, ASPP and Decoder
    The orginal artile refers to
    "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
     Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.
     (https://arxiv.org/abs/1802.02611)

    Args:
        backbone (str): backbone name, currently support Xception65, Resnet101_vd. Default Resnet101_vd.

        num_classes (int): the unique number of target classes. Default 2.

        output_stride (int): the ratio of input size and final feature size. Default 16.

        backbone_indices (tuple): two values in the tuple indicte the indices of output of backbone.
                        the first index will be taken as a low-level feature in Deconder component;
                        the second one will be taken as input of ASPP component.
                        Usually backbone consists of four downsampling stage, and return an output of
                        each stage, so we set default (0, 3), which means taking feature map of the first
                        stage in backbone as low-level feature used in Decoder, and feature map of the fourth
                        stage as input of ASPP.

        backbone_channels (tuple): the same length with "backbone_indices". It indicates the channels of corresponding index.

        ignore_index (int): the value of ground-truth mask would be ignored while doing evaluation. Default 255.

        using_sep_conv (bool): a bool value indicates whether using separable convolutions
                        in ASPP and Decoder components. Default True.
        pretrained_model (str): the pretrained_model path of backbone.
    """

    def __init__(self,
                 backbone,
                 num_classes=2,
                 output_stride=16,
                 backbone_indices=(0, 3),
                 backbone_channels=(256, 2048),
                 ignore_index=255,
                 using_sep_conv=True,
                 pretrained_model=None):

        super(DeepLabV3P, self).__init__()

        self.backbone = manager.BACKBONES[backbone](output_stride=output_stride)
        self.aspp = ASPP(output_stride, backbone_channels[1], using_sep_conv)
        self.decoder = Decoder(num_classes, backbone_channels[0], using_sep_conv)
        self.ignore_index = ignore_index
        self.EPS = 1e-5
        self.backbone_indices = backbone_indices
        self.init_weight(pretrained_model)

    def forward(self, input, label=None):

        _, feat_list = self.backbone(input)
        low_level_feat = feat_list[self.backbone_indices[0]]
        x = feat_list[self.backbone_indices[1]]
        x = self.aspp(x)
        logit = self.decoder(x, low_level_feat)
        logit = fluid.layers.resize_bilinear(logit, input.shape[2:])

        if self.training:
            return self._get_loss(logit, label)
        else:
            score_map = fluid.layers.softmax(logit, axis=1)
            score_map = fluid.layers.transpose(score_map, [0, 2, 3, 1])
            pred = fluid.layers.argmax(score_map, axis=3)
            pred = fluid.layers.unsqueeze(pred, axes=[3])
            return pred, score_map

    def init_weight(self, pretrained_model=None):
        """
        Initialize the parameters of model parts.
        Args:
            pretrained_model ([str], optional): the pretrained_model path of backbone. Defaults to None.
        """
        if pretrained_model is not None:
            if os.path.exists(pretrained_model):
                utils.load_pretrained_model(self.backbone, pretrained_model)
                # utils.load_pretrained_model(self, pretrained_model)
                # for param in self.backbone.parameters():
                #     param.stop_gradient = True

    def _get_loss(self, logit, label):
        """
        compute forward loss of the model

        Args:
            logit (tensor): the logit of model output
            label (tensor): ground truth

        Returns:
            avg_loss (tensor): forward loss
        """
        logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
        label = fluid.layers.transpose(label, [0, 2, 3, 1])
        mask = label != self.ignore_index
        mask = fluid.layers.cast(mask, 'float32')
        loss, probs = fluid.layers.softmax_with_cross_entropy(
            logit,
            label,
            ignore_index=self.ignore_index,
            return_softmax=True,
            axis=-1)

        loss = loss * mask
        avg_loss = fluid.layers.mean(loss) / (
                fluid.layers.mean(mask) + self.EPS)

        label.stop_gradient = True
        mask.stop_gradient = True

        return avg_loss


def build_aspp(output_stride, using_sep_conv):
    return ASPP(output_stride=output_stride, using_sep_conv=using_sep_conv)


def build_decoder(num_classes, using_sep_conv):
    return Decoder(num_classes, using_sep_conv=using_sep_conv)

@manager.MODELS.add_component
def deeplabv3p_resnet101_vd(*args, **kwargs):
    pretrained_model = None
    return DeepLabV3P(backbone='ResNet101_vd', pretrained_model=pretrained_model, **kwargs)

@manager.MODELS.add_component
def deeplabv3p_resnet101_vd_os8(*args, **kwargs):
    pretrained_model = None
    return DeepLabV3P(backbone='ResNet101_vd', output_stride=8, pretrained_model=pretrained_model, **kwargs)

@manager.MODELS.add_component
def deeplabv3p_resnet50_vd(*args, **kwargs):
    pretrained_model = None
    return DeepLabV3P(backbone='ResNet50_vd', pretrained_model=pretrained_model, **kwargs)

@manager.MODELS.add_component
def deeplabv3p_resnet50_vd_os8(*args, **kwargs):
    pretrained_model = None
    return DeepLabV3P(backbone='ResNet50_vd', output_stride=8, pretrained_model=pretrained_model, **kwargs)

@manager.MODELS.add_component
def deeplabv3p_xception65_deeplab(*args, **kwargs):
    pretrained_model = None
    return DeepLabV3P(backbone='Xception65_deeplab',
                      pretrained_model=pretrained_model,
                      backbone_indices=(0, 1),
                      backbone_channels=(128, 2048),
                      **kwargs)

@manager.MODELS.add_component
def deeplabv3p_mobilenetv3_large(*args, **kwargs):
    pretrained_model = None
    return DeepLabV3P(backbone='MobileNetV3_large_x1_0',
                      pretrained_model=pretrained_model,
                      backbone_indices=(0, 3),
                      backbone_channels=(24, 160),
                      **kwargs)

@manager.MODELS.add_component
def deeplabv3p_mobilenetv3_small(*args, **kwargs):
    pretrained_model = None
    return DeepLabV3P(backbone='MobileNetV3_small_x1_0',
                      pretrained_model=pretrained_model,
                      backbone_indices=(0, 3),
                      backbone_channels=(16, 96),
                      **kwargs)

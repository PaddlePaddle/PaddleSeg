# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils

__all__ = ['PanopticDeepLab']


@manager.MODELS.add_component
class PanopticDeepLab(nn.Layer):
    """
    The PanopticDeeplab implementation based on PaddlePaddle.

    The original article refers to
     Bowen Cheng, et, al. "Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation"
     (https://arxiv.org/abs/1911.10194)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone network, currently support Resnet50_vd/Resnet101_vd/Xception65.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
           Default: (2, 1, 0, 3).
        aspp_ratios (tuple, optional): The dilation rate using in ASSP module.
            If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            If output_stride=8, aspp_ratios is (1, 12, 24, 36).
            Default: (1, 6, 12, 18).
        aspp_out_channels (int, optional): The output channels of ASPP module. Default: 256.
        decoder_channels (int, optional): The channels of Decoder. Default: 256.
        low_level_channels_projects (list, opitonal). The channels of low level features to output. Defualt: None.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(2, 1, 0, 3),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256,
                 decoder_channels=256,
                 low_level_channels_projects=None,
                 align_corners=False,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = PanopticDeepLabHead(
            num_classes, backbone_indices, backbone_channels, aspp_ratios,
            aspp_out_channels, decoder_channels, align_corners,
            low_level_channels_projects, **kwargs)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction, with special handling to offset.

            Args:
                pred (dict): stores all output of the segmentation model.
                input_shape (tuple): spatial resolution of the desired shape.

            Returns:
                result (OrderedDict): upsampled dictionary.
            """
        # Override upsample method to correctly handle `offset`
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(
                pred[key],
                size=input_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            if 'offset' in key:
                if input_shape[0] % 2 == 0:
                    scale = input_shape[0] // pred[key].shape[2]
                else:
                    scale = (input_shape[0] - 1) // (pred[key].shape[2] - 1)
                out *= scale
            result[key] = out
        return result

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_dict = self.head(feat_list)
        results = self._upsample_predictions(logit_dict, x.shape[-2:])

        # return results
        logit_list = [results['semantic'], results['center'], results['offset']]
        return logit_list
        # return [results['semantic']]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class PanopticDeepLabHead(nn.Layer):
    """
    The DeepLabV3PHead implementation based on PaddlePaddle.

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): Two values in the tuple indicate the indices of output of backbone.
            the first index will be taken as a low-level feature in Decoder component;
            the second one will be taken as input of ASPP component.
            Usually backbone consists of four downsampling stage, and return an output of
            each stage. If we set it as (0, 3), it means taking feature map of the first
            stage in backbone as low-level feature used in Decoder, and feature map of the fourth
            stage as input of ASPP.
        backbone_channels (tuple): The same length with "backbone_indices". It indicates the channels of corresponding index.
        aspp_ratios (tuple): The dilation rates using in ASSP module.
        aspp_out_channels (int): The output channels of ASPP module.
        decoder_channels (int, optional): The channels of Decoder. Default: 256.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        low_level_channels_projects (list, opitonal). The channels of low level features to output. Defualt: None.
    """

    def __init__(self, num_classes, backbone_indices, backbone_channels,
                 aspp_ratios, aspp_out_channels, decoder_channels,
                 align_corners, low_level_channels_projects, **kwargs):
        super().__init__()
        self.semantic_decoder = SinglePanopticDeepLabDecoder(
            backbone_indices=backbone_indices,
            backbone_channels=backbone_channels,
            aspp_ratios=aspp_ratios,
            aspp_out_channels=aspp_out_channels,
            decoder_channels=decoder_channels,
            align_corners=align_corners,
            low_level_channels_projects=low_level_channels_projects)
        self.semantic_head = SinglePanopticDeepLabHead(
            num_classes=[num_classes],
            decoder_channels=decoder_channels,
            head_channels=decoder_channels,
            class_key=['semantic'])
        self.instance_decoder = SinglePanopticDeepLabDecoder(
            backbone_indices=backbone_indices,
            backbone_channels=backbone_channels,
            aspp_ratios=aspp_ratios,
            aspp_out_channels=kwargs['instance_aspp_out_channels'],
            decoder_channels=kwargs['instance_decoder_channels'],
            align_corners=align_corners,
            low_level_channels_projects=kwargs[
                'instance_low_level_channels_projects'])
        self.instance_head = SinglePanopticDeepLabHead(
            num_classes=kwargs['instance_num_classes'],
            decoder_channels=kwargs['instance_decoder_channels'],
            head_channels=kwargs['instance_head_channels'],
            class_key=kwargs['instance_class_key'])

    def forward(self, features):
        # pred = OrdereDict()
        pred = {}

        # Semantic branch
        semantic = self.semantic_decoder(features)
        semantic = self.semantic_head(semantic)
        for key in semantic.keys():
            pred[key] = semantic[key]

        # Instance branch
        instance = self.instance_decoder(features)
        instance = self.instance_head(instance)
        for key in instance.keys():
            pred[key] = instance[key]

        return pred


class SeparableConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self.depthwise_conv = layers.ConvBNReLU(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)
        self.piontwise_conv = layers.ConvBNReLU(
            in_channels, out_channels, kernel_size=1, groups=1, bias_attr=False)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class ASPPModule(nn.Layer):
    """
    Atrous Spatial Pyramid Pooling.

    Args:
        aspp_ratios (tuple): The dilation rate using in ASSP module.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        use_sep_conv (bool, optional): If using separable conv in ASPP module. Default: False.
        image_pooling (bool, optional): If augmented with image-level features. Default: False
        drop_rate (float, optional): The drop rate. Default: 0.1.
    """

    def __init__(self,
                 aspp_ratios,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_sep_conv=False,
                 image_pooling=False,
                 drop_rate=0.1):
        super().__init__()

        self.align_corners = align_corners
        self.aspp_blocks = nn.LayerList()

        for ratio in aspp_ratios:
            if use_sep_conv and ratio > 1:
                conv_func = SeparableConvBNReLU
            else:
                conv_func = layers.ConvBNReLU

            block = conv_func(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1 if ratio == 1 else 3,
                dilation=ratio,
                padding=0 if ratio == 1 else ratio,
                bias_attr=False)
            self.aspp_blocks.append(block)

        out_size = len(self.aspp_blocks)

        if image_pooling:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2D(output_size=(1, 1)),
                layers.ConvBNReLU(
                    in_channels, out_channels, kernel_size=1, bias_attr=False))
            out_size += 1
        self.image_pooling = image_pooling

        self.conv_bn_relu = layers.ConvBNReLU(
            in_channels=out_channels * out_size,
            out_channels=out_channels,
            kernel_size=1,
            bias_attr=False)

        self.dropout = nn.Dropout(p=drop_rate)  # drop rate

    def forward(self, x):
        outputs = []
        for block in self.aspp_blocks:
            y = block(x)
            interpolate_shape = x.shape[2:]
            y = F.interpolate(
                y,
                interpolate_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            outputs.append(y)

        if self.image_pooling:
            img_avg = self.global_avg_pool(x)
            img_avg = F.interpolate(
                img_avg,
                interpolate_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            outputs.append(img_avg)

        x = paddle.concat(outputs, axis=1)
        x = self.conv_bn_relu(x)
        x = self.dropout(x)

        return x


class SinglePanopticDeepLabDecoder(nn.Layer):
    """
    The DeepLabV3PHead implementation based on PaddlePaddle.

    Args:
        backbone_indices (tuple): Two values in the tuple indicate the indices of output of backbone.
            the first index will be taken as a low-level feature in Decoder component;
            the second one will be taken as input of ASPP component.
            Usually backbone consists of four downsampling stage, and return an output of
            each stage. If we set it as (0, 3), it means taking feature map of the first
            stage in backbone as low-level feature used in Decoder, and feature map of the fourth
            stage as input of ASPP.
        backbone_channels (tuple): The same length with "backbone_indices". It indicates the channels of corresponding index.
        aspp_ratios (tuple): The dilation rates using in ASSP module.
        aspp_out_channels (int): The output channels of ASPP module.
        decoder_channels (int): The channels of decoder
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        low_level_channels_projects (list). The channels of low level features to output.
    """

    def __init__(self, backbone_indices, backbone_channels, aspp_ratios,
                 aspp_out_channels, decoder_channels, align_corners,
                 low_level_channels_projects):
        super().__init__()
        self.aspp = ASPPModule(
            aspp_ratios,
            backbone_channels[-1],
            aspp_out_channels,
            align_corners,
            use_sep_conv=False,
            image_pooling=True,
            drop_rate=0.5)
        self.backbone_indices = backbone_indices
        self.decoder_stage = len(low_level_channels_projects)
        if self.decoder_stage != len(self.backbone_indices) - 1:
            raise ValueError(
                "len(low_level_channels_projects) != len(backbone_indices) - 1, they are {} and {}"
                .format(low_level_channels_projects, backbone_indices))
        self.align_corners = align_corners

        # Transform low-level feature
        project = []
        # Fuse
        fuse = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            project.append(
                layers.ConvBNReLU(
                    backbone_channels[i],
                    low_level_channels_projects[i],
                    1,
                    bias_attr=False))
            if i == 0:
                fuse_in_channels = aspp_out_channels + low_level_channels_projects[
                    i]
            else:
                fuse_in_channels = decoder_channels + low_level_channels_projects[
                    i]
            fuse.append(
                SeparableConvBNReLU(
                    fuse_in_channels,
                    decoder_channels,
                    5,
                    padding=2,
                    bias_attr=False))
        self.project = nn.LayerList(project)
        self.fuse = nn.LayerList(fuse)

    def forward(self, feat_list):
        x = feat_list[self.backbone_indices[-1]]
        x = self.aspp(x)

        for i in range(self.decoder_stage):
            l = feat_list[self.backbone_indices[i]]
            l = self.project[i](l)
            x = F.interpolate(
                x,
                size=l.shape[-2:],
                mode='bilinear',
                align_corners=self.align_corners)
            x = paddle.concat([x, l], axis=1)
            x = self.fuse[i](x)

        return x


class SinglePanopticDeepLabHead(nn.Layer):
    """
    Decoder module of DeepLabV3P model

    Args:
        num_classes (int): The number of classes.
        decoder_channels (int): The channels of decoder.
        head_channels (int): The channels of head.
        class_key (list): The key name of output by classifier.
    """

    def __init__(self, num_classes, decoder_channels, head_channels, class_key):
        super(SinglePanopticDeepLabHead, self).__init__()
        self.num_head = len(num_classes)
        if self.num_head != len(class_key):
            raise ValueError(
                "len(num_classes) != len(class_key), they are {} and {}".format(
                    num_classes, class_key))

        classifier = []
        for i in range(self.num_head):
            classifier.append(
                nn.Sequential(
                    SeparableConvBNReLU(
                        decoder_channels,
                        head_channels,
                        5,
                        padding=2,
                        bias_attr=False),
                    nn.Conv2D(head_channels, num_classes[i], 1)))

        self.classifier = nn.LayerList(classifier)
        self.class_key = class_key

    def forward(self, x):
        pred = OrderedDict()
        # build classifier
        for i, key in enumerate(self.class_key):
            pred[key] = self.classifier[i](x)

        return pred

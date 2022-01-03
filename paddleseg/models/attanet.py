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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers

__all__ = ['AttaNet']


@manager.MODELS.add_component
class AttaNet(nn.Layer):
    """
    The AttaNet implementation based on PaddlePaddle.

    The original article refers to Song, Qi, Kangfu Mei, and Rui Huang.
    "AttaNet: Attention-Augmented Network for Fast and Accurate Scene
    Parsing." (https://arxiv.org/abs/2103.05930)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone networks.
        backbone_indices (tuple, optional): The values in the tuple indicate the indices
            of output of backbone. Default: (-1, -2, -3).
        fpn_channels (int, optional): The channels of FPN. Default: 128.
        head_resize_mode (str): The mode for F.interpolate in the head of AttaNet.
        align_corners (bool): An argument of F.interpolate. It should be set to False when
            the output size of feature is even, e.g. 1024x512, otherwise it is True,
            e.g. 769x769.  Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(-1, -2, -3),
                 fpn_channels=128,
                 head_resize_mode='nearest',
                 afm_type='AttentionFusionModule',
                 sam_type='StripAttentionModule',
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        assert len(
            backbone_indices) == 3, "The length of backbone_indices should be 3"
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = AttaNetHead(backbone_indices, backbone_channels,
                                fpn_channels, head_resize_mode, afm_type,
                                sam_type)

        self.conv_out = AttaNetOutput(fpn_channels, fpn_channels, num_classes)
        self.conv_aux1 = AttaNetOutput(fpn_channels, fpn_channels // 2,
                                       num_classes)
        self.conv_aux2 = AttaNetOutput(fpn_channels, fpn_channels // 2,
                                       num_classes)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)

        out, aux1, aux2 = self.head(feat_list)

        out = self.conv_out(out)
        logit_list = [out]

        if self.training:
            aux1 = self.conv_aux1(aux1)
            aux2 = self.conv_aux2(aux2)
            logit_list.extend([aux1, aux2])

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


class AttaNetHead(nn.Layer):
    """
    The head of AttaNet.

    Args:
        backbone_indices (tuple): The values in the tuple indicate the indices of
            output of backbone.
        backbone_channels (tuple): The channels of the output of backbone, such as
            [f32_chan, f16_chan, f8_chan].
        fpn_channels (int): The channels of FPN. Default: 128.
        resize_mode (str): The mode for F.interpolate.
    """

    def __init__(self, backbone_indices, backbone_channels, fpn_channels,
                 resize_mode, afm_type, sam_type):
        super().__init__()
        assert len(backbone_indices) == 3
        assert len(backbone_channels) == 3

        self.backbone_indices = backbone_indices
        f32_chan, f16_chan, _ = backbone_channels

        print('afm_type:', afm_type)
        if afm_type == 'AttentionFusionModule':
            self.afm = AttentionFusionModule(f16_chan, f32_chan, fpn_channels,
                                             resize_mode)
        else:
            self.conv_f32 = layers.ConvBNReLU(
                f32_chan,
                fpn_channels,
                kernel_size=3,
                padding=1,
                bias_attr=False)
            afm_class = getattr(layers, afm_type)
            self.afm = afm_class(f16_chan, fpn_channels, fpn_channels, 3,
                                 resize_mode)

        print('sam_type:', sam_type)
        if sam_type is not None:
            self.sam = eval(sam_type)(fpn_channels, fpn_channels)
        self.conv_f16_up = layers.ConvBNReLU(
            fpn_channels,
            fpn_channels,
            kernel_size=3,
            padding=1,
            bias_attr=False)

        self.resize_mode = resize_mode
        init_weight(self)

    def forward(self, feat_list):
        feat32 = feat_list[self.backbone_indices[0]]
        feat16 = feat_list[self.backbone_indices[1]]
        feat8 = feat_list[self.backbone_indices[2]]

        if hasattr(self, 'conv_f32'):
            feat32 = self.conv_f32(feat32)

        feat16_sum = self.afm(feat16, feat32)

        if hasattr(self, 'sam'):
            feat16_sum = self.sam(feat16_sum)

        feat16_up = F.interpolate(
            feat16_sum, paddle.shape(feat8)[2:], mode=self.resize_mode)
        feat16_up = self.conv_f16_up(feat16_up)

        logit_list = [feat16_up, feat16_sum, feat8]  # x8, x16, x8
        return logit_list


class AttentionFusionModule(nn.Layer):
    """
    The attention fusion module.

    Args:
        f16_chan (int): The channel of feat16 tensor.
        f32_chan (int): The channel of feat32 tensor.
        out_chan (int): The channel of output tensor.
        resize_mode (str): The mode for F.interpolate.
    """

    def __init__(self, f16_chan, f32_chan, out_chan, resize_mode):
        super().__init__()
        self.conv_f32 = layers.ConvBNReLU(
            f32_chan, out_chan, kernel_size=3, padding=1, bias_attr=False)
        self.conv_f16 = layers.ConvBNReLU(
            f16_chan, out_chan, kernel_size=3, padding=1, bias_attr=False)
        self.conv_f16_sum = layers.ConvBNReLU(
            out_chan, out_chan, kernel_size=3, padding=1, bias_attr=False)

        self.conv_cat = layers.ConvBNReLU(
            out_chan + f32_chan, out_chan, kernel_size=1, bias_attr=False)
        self.conv_atten = nn.Conv2D(
            out_chan, out_chan, kernel_size=1, bias_attr=False)
        self.bn_atten = nn.BatchNorm2D(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

        self.resize_mode = resize_mode
        init_weight(self)

    def get_atten(self, feat16, feat32):
        feat32_up = F.interpolate(
            feat32, paddle.shape(feat16)[2:], mode=self.resize_mode)
        fcat = paddle.concat([feat16, feat32_up], axis=1)
        fcat = self.conv_cat(fcat)

        atten = F.adaptive_avg_pool2d(fcat, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        return atten

    def forward(self, feat16, feat32):
        feat16 = self.conv_f16(feat16)
        atten = self.get_atten(feat16, feat32)

        feat32 = self.conv_f32(feat32)
        feat32 = feat32 * atten
        feat32_up = F.interpolate(
            feat32, paddle.shape(feat16)[2:], mode=self.resize_mode)

        feat16 = feat16 * (1 - atten)

        feat16_sum = feat16 + feat32_up
        feat16_sum = self.conv_f16_sum(feat16_sum)
        return feat16_sum


class StripAttentionModule(nn.Layer):
    """
    The strip attention module.

    Args:
        in_chan (int): The channel of input tensor.
        out_chan (int): The channel of output tensor.
    """

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv1 = layers.ConvBNReLU(
            in_chan, 64, kernel_size=1, bias_attr=False)
        self.conv2 = layers.ConvBNReLU(
            in_chan, 64, kernel_size=1, bias_attr=False)
        self.conv3 = layers.ConvBNReLU(
            in_chan, out_chan, kernel_size=1, bias_attr=False)
        self.softmax = nn.Softmax(axis=1)

        init_weight(self)

    def forward(self, x):
        height = paddle.shape(x)[2]
        width = paddle.shape(x)[3]
        # TODO: use paddle.shape to support dynamic shape
        height_int = int(x.shape[2])

        q = self.conv2(x)
        q = paddle.flatten(q, start_axis=2, stop_axis=3)  # n * c' * hw

        k = self.conv1(x)  # n * c' * h *w
        k = F.avg_pool2d(k, [height_int, 1])  # n * c' * 1 * w
        k = paddle.flatten(k, start_axis=2, stop_axis=3)  # n * c' * w
        k = paddle.transpose(k, [0, 2, 1])  # n * w * c'

        v = self.conv3(x)
        v = F.avg_pool2d(v, [height_int, 1])  # n * c * 1 * w
        v = paddle.flatten(v, start_axis=2, stop_axis=3)  # n * c * w

        atten_map = paddle.bmm(k, q)  # n * w * hw
        atten_x = paddle.bmm(v, atten_map)  # n * c * hw
        atten_x = paddle.reshape(atten_x,
                                 [0, 0, height, width])  # n * c * h * w
        out = x + atten_x
        return out


class AttaNetOutput(nn.Layer):
    """
    This module outputs the results of segmentation.

    Args:
        in_chan (int): The channel of input tensor.
        mid_chan (int): The channel of the middle tensor.
        out_chan (int): The channel of output tensor.
    """

    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            in_chan, mid_chan, kernel_size=3, padding=1, bias_attr=False)
        self.dropout = nn.Dropout2D(0.1)
        self.conv_out = nn.Conv2D(
            mid_chan, n_classes, kernel_size=1, bias_attr=False)
        init_weight(self)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        return x


def init_weight(input_layer):
    """
    Init the weight and bias of conv2d layers.
    """
    assert isinstance(input_layer,
                      nn.Layer), "The input_layer should be nn.Layer"
    for sublayer in input_layer.sublayers():
        if isinstance(sublayer, nn.Conv2D):
            param_init.kaiming_normal_init(sublayer.weight)
            if sublayer.bias is not None:
                param_init.constant_init(sublayer.bias, value=0.0)

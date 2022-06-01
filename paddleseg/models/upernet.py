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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers


@manager.MODELS.add_component
class UPerNet(nn.Layer):
    """
    The UPerNet implementation based on PaddlePaddle.

    The original article refers to
    Tete Xiao, et, al. "Unified Perceptual Parsing for Scene Understanding"
    (https://arxiv.org/abs/1807.10221).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network, currently support Resnet50/101.
        backbone_indices (tuple): Four values in the tuple indicate the indices of output of backbone.
        channels (int): The channels of inter layers. Default: 512.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: False.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        dropout_prob (float): Dropout ratio for upernet head. Default: 0.1.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices,
                 channels=512,
                 enable_auxiliary_loss=False,
                 align_corners=False,
                 dropout_prob=0.1,
                 pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        self.in_channels = [
            self.backbone.feat_channels[i] for i in backbone_indices
        ]
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.enable_auxiliary_loss = enable_auxiliary_loss

        fpn_inplanes = [
            self.backbone.feat_channels[i] for i in backbone_indices
        ]
        self.head = UPerNetHead(
            num_classes=num_classes,
            fpn_inplanes=fpn_inplanes,
            dropout_prob=dropout_prob,
            channels=channels,
            enable_auxiliary_loss=self.enable_auxiliary_loss)
        self.init_weight()

    def forward(self, x):
        feats = self.backbone(x)
        feats = [feats[i] for i in self.backbone_indices]
        logit_list = self.head(feats)
        logit_list = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class UPerNetHead(nn.Layer):
    def __init__(self,
                 num_classes,
                 fpn_inplanes,
                 channels,
                 dropout_prob=0.1,
                 enable_auxiliary_loss=False,
                 align_corners=True):
        super(UPerNetHead, self).__init__()
        self.align_corners = align_corners
        self.ppm = layers.PPModule(
            in_channels=fpn_inplanes[-1],
            out_channels=channels,
            bin_sizes=(1, 2, 3, 6),
            dim_reduction=True,
            align_corners=True)
        self.enable_auxiliary_loss = enable_auxiliary_loss
        self.lateral_convs = nn.LayerList()
        self.fpn_convs = nn.LayerList()

        for fpn_inplane in fpn_inplanes[:-1]:
            self.lateral_convs.append(
                layers.ConvBNReLU(fpn_inplane, channels, 1))
            self.fpn_convs.append(
                layers.ConvBNReLU(
                    channels, channels, 3, bias_attr=False))

        if self.enable_auxiliary_loss:
            self.aux_head = layers.AuxLayer(
                fpn_inplanes[2],
                fpn_inplanes[2],
                num_classes,
                dropout_prob=dropout_prob)

        self.fpn_bottleneck = layers.ConvBNReLU(
            len(fpn_inplanes) * channels, channels, 3, padding=1)

        self.conv_last = nn.Sequential(
            layers.ConvBNReLU(
                len(fpn_inplanes) * channels, channels, 3, bias_attr=False),
            nn.Conv2D(
                channels, num_classes, kernel_size=1))
        self.conv_seg = nn.Conv2D(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))

        laterals.append(self.ppm(inputs[-1]))
        fpn_levels = len(laterals)
        for i in range(fpn_levels - 1, 0, -1):
            prev_shape = paddle.shape(laterals[i - 1])
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = []
        for i in range(fpn_levels - 1):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))
        fpn_outs.append(laterals[-1])

        for i in range(fpn_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=paddle.shape(fpn_outs[0])[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fuse_out = paddle.concat(fpn_outs, axis=1)
        x = self.fpn_bottleneck(fuse_out)

        x = self.conv_seg(x)
        logits_list = [x]
        if self.enable_auxiliary_loss:
            aux_out = self.aux_head(inputs[2])
            logits_list.append(aux_out)
            return logits_list
        else:
            return logits_list

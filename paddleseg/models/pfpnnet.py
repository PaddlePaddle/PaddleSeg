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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.models import layers
from paddleseg.cvlibs import manager
from paddleseg.utils import utils


@manager.MODELS.add_component
class PFPNNet(nn.Layer):
    """
    The Panoptic Feature Pyramid Networks implementation based on PaddlePaddle.

    The original article refers to
    Alexander Kirillov, Ross Girshick, Kaiming He, Piotr Dollár, et al. "Panoptic Feature Pyramid Networks"
    (https://arxiv.org/abs/1901.02446)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network, currently support Resnet50/101.
        backbone_indices (tuple): Four values in the tuple indicate the indices of output of backbone.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: False.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """
    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices,
                 channels,
                 enable_auxiliary_loss=False,
                 align_corners=False,
                 dropout_ratio=0.1,
                 fpn_inplanes=[256, 512, 1024, 2048],
                 pretrained=None):
        super(PFPNNet, self).__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        self.in_channels = [
            self.backbone.feat_channels[i] for i in backbone_indices
        ]
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.enable_auxiliary_loss = enable_auxiliary_loss

        self.head = PFPNHead(num_class=num_classes,
                             fpn_inplanes=fpn_inplanes,
                             dropout_ratio=dropout_ratio,
                             channels=channels,
                             fpn_dim=channels,
                             enable_auxiliary_loss=self.enable_auxiliary_loss)
        self.init_weight()

    def forward(self, x):
        feats = self.backbone(x)
        feats = [feats[i] for i in self.backbone_indices]
        logit_list = self.head(feats)
        return [
            F.interpolate(logit,
                          paddle.shape(x)[2:],
                          mode='bilinear',
                          align_corners=self.align_corners)
            for logit in logit_list
        ]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class PFPNHead(nn.Layer):
    """
    The PFPNHead implementation.

    Args:
        inplane (int): Input channels of PPM module.
        num_class (int): The unique number of target classes.
        fpn_inplanes (list): The feature channels from backbone.
        fpn_dim (int, optional): The input channels of FPN module. Default: 512.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: False.
    """
    def __init__(self,
                 num_class,
                 fpn_inplanes,
                 channels,
                 dropout_ratio=0.1,
                 fpn_dim=256,
                 enable_auxiliary_loss=False,
                 align_corners=False):
        super(PFPNHead, self).__init__()
        self.enable_auxiliary_loss = enable_auxiliary_loss
        self.align_corners = align_corners
        self.lateral_convs = nn.LayerList()
        self.fpn_out = nn.LayerList()

        for fpn_inplane in fpn_inplanes:
            self.lateral_convs.append(
                nn.Sequential(nn.Conv2D(fpn_inplane, fpn_dim, 1),
                              layers.SyncBatchNorm(fpn_dim), nn.ReLU()))
            self.fpn_out.append(
                nn.Sequential(
                    layers.ConvBNReLU(fpn_dim, fpn_dim, 3, bias_attr=False)))

        self.scale_heads = nn.LayerList()
        for index in range(len(fpn_inplanes)):
            head_length = max(
                1, int(np.log2(fpn_inplanes[index]) - np.log2(fpn_inplanes[0])))
            scale_head = nn.LayerList()
            for head_index in range(head_length):
                scale_head.append(
                    layers.ConvBNReLU(
                        fpn_dim,
                        channels,
                        3,
                        padding=1,
                    ))
                if fpn_inplanes[index] != fpn_inplanes[0]:
                    scale_head.append(
                        nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

        if dropout_ratio:
            self.dropout = nn.Dropout2D(dropout_ratio)
            if self.enable_auxiliary_loss:
                self.dsn = nn.Sequential(
                    layers.ConvBNReLU(fpn_inplanes[2],
                                      fpn_inplanes[2],
                                      3,
                                      padding=1), nn.Dropout2D(dropout_ratio),
                    nn.Conv2D(fpn_inplanes[2], num_class, kernel_size=1))
        else:
            self.dropout = None
            if self.enable_auxiliary_loss:
                self.dsn = nn.Sequential(
                    layers.ConvBNReLU(fpn_inplanes[2],
                                      fpn_inplanes[2],
                                      3,
                                      padding=1),
                    nn.Conv2D(fpn_inplanes[2], num_class, kernel_size=1))

        self.conv_last = nn.Sequential(
            layers.ConvBNReLU(len(fpn_inplanes) * fpn_dim,
                              fpn_dim,
                              3,
                              bias_attr=False),
            nn.Conv2D(fpn_dim, num_class, kernel_size=1))
        self.conv_seg = nn.Conv2D(channels, num_class, kernel_size=1)

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, conv_out):
        last_out = self.lateral_convs[-1](conv_out[-1])
        f = last_out
        fpn_feature_list = [last_out]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.lateral_convs[i](conv_x)
            prev_shape = paddle.shape(conv_x)[2:]
            f = conv_x + F.interpolate(
                f, prev_shape, mode='bilinear', align_corners=True)
            fpn_feature_list.append(self.fpn_out[i](f))

        output_size = paddle.shape(fpn_feature_list[-1])[2:]

        x = self.scale_heads[0](fpn_feature_list[-1])
        for index in range(len(self.scale_heads) - 2, 0, -1):
            x = x + F.interpolate(self.scale_heads[index](
                fpn_feature_list[index]),
                                  size=output_size,
                                  mode='bilinear',
                                  align_corners=self.align_corners)
        x = self.cls_seg(x)
        if self.enable_auxiliary_loss:
            dsn = self.dsn(conv_out[2])
            return [x, dsn]
        else:
            return [x]

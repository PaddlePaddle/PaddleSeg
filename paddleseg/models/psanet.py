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

import paddle.nn as nn
import paddle.nn.functional as F

import paddle
from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils
from paddleseg.models.ops import psa_mask_op


@manager.MODELS.add_component
class PSANet(nn.Layer):
    """
    The PSANet implementation based on PaddlePaddle.

    The original article refers to
    Zhao, Hengshuang, et al. "PSANet: Point-wise Spatial Attention Network for Scene Parsing"
    (https://hszhao.github.io/papers/eccv18_psanet.pdf).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network, currently support Resnet50/101.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        psa_type (int, optional): The type of point-wise spatial attention. 0 represents 'collect'. 1 represents 'distribute'. 2 represents 'bi-direction'. Default: 2.
        shrink_factor(int, optional): The shrink factor of point-wise spatial attention module input feature map. Default: 2.
        mask_h (int, optional): The height of point-wise spatial attention mask. The value of mask_h must be odd number. Default: 59.
        mask_w (int, optional): The weight of point-wise spatial attention mask. The value of mask_h must be odd number. Default: 59.
        normalization_factor (float, optional): The normalize factor of point-wise spatial attention module. Default: 1.0.
        psa_softmax (bool, optional): Whether use softmax after psamask operation. Default: True.
        dropout (float, optional): Dropout ratio in PSANet head and auxiliary head. Default: 0.1.
        use_psa (bool, optional): Whether use point-wise spatial attention module. Default: True.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(2, 3),
                 enable_auxiliary_loss=True,
                 psa_type=2,
                 shrink_factor=2,
                 mask_h=59,
                 mask_w=59,
                 normalization_factor=1.0,
                 psa_softmax=True,
                 dropout=0.1,
                 use_psa=True,
                 align_corners=False,
                 pretrained=None):
        super().__init__()
        self.use_psa = use_psa
        self.enable_auxiliary_loss = enable_auxiliary_loss
        self.backbone = backbone
        self.backbone_indices = backbone_indices

        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        if use_psa:
            self.psa = PSA(backbone_channels[-1], 512, psa_type, shrink_factor,
                           mask_h, mask_w, normalization_factor, psa_softmax)
        self.head = layers.AuxLayer(
            backbone_channels[-1] * (2 if use_psa else 1),
            512,
            num_classes,
            dropout_prob=dropout,
            bias_attr=False)
        if enable_auxiliary_loss:
            self.aux_head = layers.AuxLayer(
                backbone_channels[-2],
                256,
                num_classes,
                dropout_prob=dropout,
                bias_attr=False)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        feat_list = self.backbone(x)
        feat = feat_list[self.backbone_indices[-1]]
        if self.use_psa:
            feat = self.psa(feat)
        logit_list = []

        out = self.head(feat)
        logit_list.append(out)

        if self.training and self.enable_auxiliary_loss:
            aux_feat = feat_list[self.backbone_indices[-2]]
            aux_out = self.aux_head(aux_feat)
            logit_list.append(aux_out)

        return [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]


class PSA(nn.Layer):
    def __init__(self,
                 in_channels=2048,
                 mid_channels=512,
                 psa_type=2,
                 shrink_factor=2,
                 mask_h=59,
                 mask_w=59,
                 normalization_factor=1.0,
                 psa_softmax=True):
        super().__init__()
        assert psa_type in [
            0, 1, 2
        ], "Only support psa type in [0, 1, 2], but got {}.".format(psa_type)
        assert mask_h % 2 == 1 and mask_w % 2 == 1, "The value of mask_h and mask_w in point-wise spatial attention module must be odd number, but got {} and {}.".format(
            mask_h, mask_w)
        self.mid_channels = mid_channels
        self.psa_type = psa_type
        self.shrink_factor = shrink_factor
        self.mask_h = mask_h
        self.mask_w = mask_w
        self.psa_softmax = psa_softmax
        if normalization_factor is None:
            normalization_factor = mask_h * mask_w
        self.normalization_factor = normalization_factor

        self.reduce_layer = layers.ConvBNReLU(
            in_channels, mid_channels, kernel_size=1, bias_attr=False)
        self.attention = nn.Sequential(
            layers.ConvBNReLU(
                mid_channels, mid_channels, kernel_size=1, bias_attr=False),
            nn.Conv2D(
                mid_channels, mask_h * mask_w, kernel_size=1, bias_attr=False),
        )

        if psa_type == 2:
            self.reduce_p = layers.ConvBNReLU(
                in_channels, mid_channels, kernel_size=1, bias_attr=False)
            self.attention_p = nn.Sequential(
                layers.ConvBNReLU(
                    mid_channels, mid_channels, kernel_size=1, bias_attr=False),
                nn.Conv2D(
                    mid_channels,
                    mask_h * mask_w,
                    kernel_size=1,
                    bias_attr=False), )
        self.proj = layers.ConvBNReLU(
            mid_channels * (2 if psa_type == 2 else 1),
            in_channels,
            kernel_size=1,
            bias_attr=False)

    def forward(self, x):
        input_shape = paddle.shape(x)
        residual = x
        if self.psa_type in [0, 1]:
            x = self.reduce_layer(x)
            n, c, h, w = paddle.shape(x)
            h = (h - 1) // self.shrink_factor + 1
            w = (w - 1) // self.shrink_factor + 1
            x = F.interpolate(
                x, size=[h, w], mode='bilinear', align_corners=True)
            y = self.attention(x)
            y = psa_mask_op(y, self.psa_type, y.shape[0], y.shape[2],
                            y.shape[3], self.mask_h, self.mask_w,
                            self.mask_h // 2, self.mask_w // 2)
            if self.psa_softmax:
                y = F.softmax(y, axis=1)
            x = paddle.bmm(
                x.reshape([n, self.mid_channels, h * w]),
                y.reshape([n, h * w, h * w])).reshape(
                    [n, self.mid_channels, h, w]) / self.normalization_factor
        elif self.psa_type == 2:
            x_col = self.reduce_layer(x)
            x_dis = self.reduce_p(x)
            n, c, h, w = paddle.shape(x_col)
            h = (h - 1) // self.shrink_factor + 1
            w = (w - 1) // self.shrink_factor + 1
            x_col = F.interpolate(
                x_col, size=[h, w], mode='bilinear', align_corners=True)
            x_dis = F.interpolate(
                x_dis, size=[h, w], mode='bilinear', align_corners=True)

            y_col = self.attention(x_col)
            y_dis = self.attention_p(x_dis)
            y_col = psa_mask_op(y_col, 0, y_col.shape[0], y_col.shape[2],
                                y_col.shape[3], self.mask_h, self.mask_w,
                                self.mask_h // 2, self.mask_w // 2)
            y_dis = psa_mask_op(y_dis, 1, y_dis.shape[0], y_dis.shape[2],
                                y_dis.shape[3], self.mask_h, self.mask_w,
                                self.mask_h // 2, self.mask_w // 2)
            if self.psa_softmax:
                y_col = F.softmax(y_col, axis=1)
                y_dis = F.softmax(y_dis, axis=1)
            n, c, h, w = paddle.shape(x_col)
            x_col = paddle.bmm(
                x_col.reshape([n, self.mid_channels, h * w]),
                y_col.reshape([n, h * w, h * w])).reshape(
                    [n, self.mid_channels, h, w]) / self.normalization_factor
            x_dis = paddle.bmm(
                x_dis.reshape([n, self.mid_channels, h * w]),
                y_dis.reshape([n, h * w, h * w])).reshape(
                    [n, self.mid_channels, h, w]) / self.normalization_factor
            x = paddle.concat([x_dis, x_col], axis=1)
        x = self.proj(x)
        x = F.interpolate(
            x,
            size=[input_shape[2], input_shape[3]],
            mode='bilinear',
            align_corners=True)
        return paddle.concat([residual, x], axis=1)

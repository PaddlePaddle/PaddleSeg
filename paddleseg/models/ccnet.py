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

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class CCNet(nn.Layer):
    """
    The CCNet implementation based on PaddlePaddle.

    The original article refers to
    Zilong Huang, et al. "CCNet: Criss-Cross Attention for Semantic Segmentation"
    (https://arxiv.org/abs/1811.11721)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone network, currently support Resnet18_vd/Resnet34_vd/Resnet50_vd/Resnet101_vd.
        backbone_indices (tuple, list, optional): Two values in the tuple indicate the indices of output of backbone. Default: (2, 3).
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        dropout_prob (float, optional): The probability of dropout. Default: 0.0.
        recurrence (int, optional): The number of recurrent operations. Defautl: 1.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(2, 3),
                 enable_auxiliary_loss=True,
                 dropout_prob=0.0,
                 recurrence=1,
                 align_corners=False,
                 pretrained=None):
        super().__init__()
        self.enable_auxiliary_loss = enable_auxiliary_loss
        self.recurrence = recurrence
        self.align_corners = align_corners

        self.backbone = backbone
        self.backbone_indices = backbone_indices
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        if enable_auxiliary_loss:
            self.aux_head = layers.AuxLayer(
                backbone_channels[0],
                512,
                num_classes,
                dropout_prob=dropout_prob)
        self.head = RCCAModule(
            backbone_channels[1],
            512,
            num_classes,
            dropout_prob=dropout_prob,
            recurrence=recurrence)
        self.pretrained = pretrained

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = []
        output = self.head(feat_list[self.backbone_indices[-1]])
        logit_list.append(output)
        if self.training and self.enable_auxiliary_loss:
            aux_out = self.aux_head(feat_list[self.backbone_indices[-2]])
            logit_list.append(aux_out)
        return [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]


class RCCAModule(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_classes,
                 dropout_prob=0.1,
                 recurrence=1):
        super().__init__()
        inter_channels = in_channels // 4
        self.recurrence = recurrence
        self.conva = layers.ConvBNLeakyReLU(
            in_channels, inter_channels, 3, padding=1, bias_attr=False)
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = layers.ConvBNLeakyReLU(
            inter_channels, inter_channels, 3, padding=1, bias_attr=False)
        self.out = layers.AuxLayer(
            in_channels + inter_channels,
            out_channels,
            num_classes,
            dropout_prob=dropout_prob)

    def forward(self, x):
        feat = self.conva(x)
        for i in range(self.recurrence):
            feat = self.cca(feat)
        feat = self.convb(feat)
        output = self.out(paddle.concat([x, feat], axis=1))
        return output


class CrissCrossAttention(nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.q_conv = nn.Conv2D(in_channels, in_channels // 8, kernel_size=1)
        self.k_conv = nn.Conv2D(in_channels, in_channels // 8, kernel_size=1)
        self.v_conv = nn.Conv2D(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(axis=3)
        self.gamma = self.create_parameter(
            shape=(1, ), default_initializer=nn.initializer.Constant(0))
        self.inf_tensor = paddle.full(shape=(1, ), fill_value=float('inf'))

    def forward(self, x):
        b, c, h, w = paddle.shape(x)
        proj_q = self.q_conv(x)
        proj_q_h = proj_q.transpose([0, 3, 1, 2]).reshape(
            [b * w, -1, h]).transpose([0, 2, 1])
        proj_q_w = proj_q.transpose([0, 2, 1, 3]).reshape(
            [b * h, -1, w]).transpose([0, 2, 1])

        proj_k = self.k_conv(x)
        proj_k_h = proj_k.transpose([0, 3, 1, 2]).reshape([b * w, -1, h])
        proj_k_w = proj_k.transpose([0, 2, 1, 3]).reshape([b * h, -1, w])

        proj_v = self.v_conv(x)
        proj_v_h = proj_v.transpose([0, 3, 1, 2]).reshape([b * w, -1, h])
        proj_v_w = proj_v.transpose([0, 2, 1, 3]).reshape([b * h, -1, w])

        energy_h = (paddle.bmm(proj_q_h, proj_k_h) + self.Inf(b, h, w)).reshape(
            [b, w, h, h]).transpose([0, 2, 1, 3])
        energy_w = paddle.bmm(proj_q_w, proj_k_w).reshape([b, h, w, w])
        concate = self.softmax(paddle.concat([energy_h, energy_w], axis=3))

        attn_h = concate[:, :, :, 0:h].transpose([0, 2, 1, 3]).reshape(
            [b * w, h, h])
        attn_w = concate[:, :, :, h:h + w].reshape([b * h, w, w])
        out_h = paddle.bmm(proj_v_h, attn_h.transpose([0, 2, 1])).reshape(
            [b, w, -1, h]).transpose([0, 2, 3, 1])
        out_w = paddle.bmm(proj_v_w, attn_w.transpose([0, 2, 1])).reshape(
            [b, h, -1, w]).transpose([0, 2, 1, 3])
        return self.gamma * (out_h + out_w) + x

    def Inf(self, B, H, W):
        return -paddle.tile(
            paddle.diag(paddle.tile(self.inf_tensor, [H]), 0).unsqueeze(0),
            [B * W, 1, 1])

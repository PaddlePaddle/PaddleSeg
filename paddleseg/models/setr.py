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

import paddle.nn as nn
import paddle.nn.functional as F

import paddle
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class SegmentationTransformer(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(9, 14, 19, 23),
                 head='naive',
                 align_corners=False,
                 pretrained=None):

        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        if head.lower() == 'naive':
            self.head = NaiveHead(
                num_classes=self.num_classes,
                backbone_indices=backbone_indices,
                in_channels=self.backbone.embed_dim)
        elif head.lower() == 'pup':
            self.head = PUPHead(
                num_classes=self.num_classes,
                backbone_indices=backbone_indices,
                in_channels=self.backbone.embed_dim)
        elif head.lower() == 'mla':
            self.head = MLAHead()
        else:
            raise RuntimeError(
                'Unsupported segmentation head type {}. Only naive/pup/mla is valid.'
                .format(head))

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        x_shape = paddle.shape(x)
        feats, _shape = self.backbone(x)
        logits = self.head(feats, _shape)
        return [
            F.interpolate(
                _logit,
                x_shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for _logit in logits
        ]


class NaiveHead(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone_indices,
                 in_channels,
                 learning_rate_coeff=20):
        super().__init__()

        self.cls_head_norm = nn.LayerNorm(
            normalized_shape=in_channels, epsilon=1e-6)
        self.cls_head = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=in_channels, out_channels=256, kernel_size=1),
            nn.Conv2D(in_channels=256, out_channels=num_classes, kernel_size=1))

        aux_head_nums = len(backbone_indices) - 1
        self.aux_head_norms = nn.LayerList(
            [nn.LayerNorm(normalized_shape=in_channels, epsilon=1e-6)
             ] * aux_head_nums)
        self.aux_heads = nn.LayerList([
            nn.Sequential(
                layers.ConvBNReLU(
                    in_channels=in_channels, out_channels=256, kernel_size=1),
                nn.Conv2D(
                    in_channels=256, out_channels=num_classes, kernel_size=1))
        ] * aux_head_nums)

        self.in_channels = in_channels
        self.learning_rate_coeff = learning_rate_coeff
        self.backbone_indices = backbone_indices
        self.init_weight()

    def init_weight(self):
        for _param in self.parameters():
            _param.optimize_attr['learning_rate'] = self.learning_rate_coeff

        for layer in self.sublayers():
            if isinstance(layer, nn.LayerNorm):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)

    def forward(self, x, _shape):
        logits = []
        feat = x[self.backbone_indices[-1]]
        feat = self.cls_head_norm(feat).transpose([0, 2, 1]).reshape(
            [0, self.in_channels, _shape[2], _shape[3]])

        logits.append(self.cls_head(feat))

        if self.training:
            for idx, _head in enumerate(self.aux_heads):
                feat = x[self.backbone_indices[idx]]
                feat = self.aux_head_norms[idx](feat).transpose(
                    [0, 2,
                     1]).reshape([0, self.in_channels, _shape[2], _shape[3]])
                logits.append(_head(feat))

        return logits


class PUPHead(nn.Layer):
    def __init__(self, num_classes, backbone_indices, in_channels):
        super().__init__()

        inter_channels = in_channels // 4

        self.cls_head = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=1,
                stride=1), nn.Upsample(scale_factor=2),
            layers.ConvBNReLU(
                in_channels=inter_channels,
                out_channels=inter_channels,
                kernel_size=1,
                stride=1), nn.Upsample(scale_factor=2),
            layers.ConvBNReLU(
                in_channels=inter_channels,
                out_channels=inter_channels,
                kernel_size=1,
                stride=1), nn.Upsample(scale_factor=2),
            nn.Conv2D(
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=1))

        aux_head_nums = len(backbone_indices) - 1
        self.aux_heads = nn.LayerList([
            layers.AuxLayer(
                in_channels=in_channels,
                inter_channels=in_channels,
                out_channels=num_classes)
        ] * aux_head_nums)
        self.backbone_indices = backbone_indices

        self.init_weight()

    def init_weight(self):
        # Do not initialize auxiliary layer.
        for layer in self.cls_head.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)

    def forward(self, x):
        logits = []
        feat = x[self.backbone_indices[-1]]
        logits.append(self.cls_head(feat))

        if self.training:
            for idx, _head in enumerate(self.aux_heads):
                feat = x[self.backbone_indices[idx]]
                logits.append(_head(feat))

        return logits


class MLAHead(nn.Layer):
    def __init__(self):
        ...

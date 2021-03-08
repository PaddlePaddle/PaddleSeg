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

from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class SegmentationTransformer(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 head='naive',
                 align_corners=False,
                 pretrained=None):

        super().__init__()
        self.backbone = backbone
        step = self.backbone.depth // 4
        self.backbone_indices = list(range(self.backbone.depth - 1, 0, -step))
        self.num_classes = num_classes

        if head == 'naive':
            inter_channels = self.backbone.embed_dim
            self.head = NaiveHead(
                num_classes=self.num_classes,
                in_channels=self.backbone.embed_dim,
                inter_channels=inter_channels)
        elif head == 'pup':
            inter_channels = self.backbone.embed_dim // 4
            self.head = PUPHead(
                num_classes=self.num_classes,
                in_channels=self.backbone.embed_dim,
                inter_channels=inter_channels)
        elif head == 'mla':
            self.head = MLAHead()
        else:
            raise RuntimeError()

        self.aux_head_1 = layers.AuxLayer(
            in_channels=self.backbone.embed_dim,
            inter_channels=self.backbone.embed_dim,
            out_channels=self.num_classes)
        self.aux_head_2 = layers.AuxLayer(
            in_channels=self.backbone.embed_dim,
            inter_channels=self.backbone.embed_dim,
            out_channels=self.num_classes)
        self.aux_head_3 = layers.AuxLayer(
            in_channels=self.backbone.embed_dim,
            inter_channels=self.backbone.embed_dim,
            out_channels=self.num_classes)
        self.aux_head_4 = layers.AuxLayer(
            in_channels=self.backbone.embed_dim,
            inter_channels=self.backbone.embed_dim,
            out_channels=self.num_classes)

        self.cls_head = nn.Conv2D(
            in_channels=inter_channels,
            out_channels=num_classes,
            kernel_size=1,
            stride=1)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        feats = self.backbone(x)
        cls_res = []
        cls_res.append(self.cls_head(self.head(feats[-1])))
        cls_res.append(self.aux_head_1(feats[self.backbone_indices[0]]))
        cls_res.append(self.aux_head_2(feats[self.backbone_indices[1]]))
        cls_res.append(self.aux_head_3(feats[self.backbone_indices[2]]))
        cls_res.append(self.aux_head_4(feats[self.backbone_indices[3]]))
        return [F.interpolate(_cls_res, x.shape[2:]) for _cls_res in cls_res]


class NaiveHead(nn.Layer):
    def __init__(self, num_classes, in_channels, inter_channels):
        super().__init__()
        self.conv_1 = layers.ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=1,
            stride=1)

        self.init_weight()

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)

    def forward(self, x):
        return self.conv_1(x)


class PUPHead(nn.Layer):
    def __init__(self, num_classes, in_channels, inter_channels):
        super().__init__()

        self.convs = nn.Sequential(
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
                stride=1), nn.Upsample(scale_factor=2))

        self.init_weight()

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)

    def forward(self, x):
        return self.convs(x)


class MLAHead(nn.Layer):
    def __init__(self):
        ...

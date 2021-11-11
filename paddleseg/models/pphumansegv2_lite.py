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

from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.utils import utils

__all__ = ['PPHumanSegV1Lite']


@manager.MODELS.add_component
class PPHumanSegV1Lite(nn.Layer):
    "A self-developed ultra lightweight model from PaddleSeg, is suitable for real-time scene segmentation on web or mobile terminals."

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(0, -1),
                 pretrained=None,
                 align_corners=False):
        super().__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.align_corners = align_corners

        self.backbone = backbone
        self.backbone_indices = backbone_indices
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = PPHumanSegV1LiteHead(
            num_classes,
            backbone_indices=backbone_indices,
            backbone_channels=backbone_channels,
            align_corners=align_corners)

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
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)


class PPHumanSegV1LiteHead(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone_indices,
                 backbone_channels,
                 align_corners=False):
        super().__init__()
        self.conv_bn_relu = _ConvBNReLU(backbone_channels[0], 18, 1, 1, 0)

        self.depthwise_separable0 = _SeparableConvBNReLU(
            backbone_channels[1], 64, 3, stride=1)
        self.depthwise_separable1 = _SeparableConvBNReLU(82, 64, 3, stride=1)
        self.depthwise_separable2 = _SeparableConvBNReLU(
            64, num_classes, 3, stride=1)

        self.backbone_indices = backbone_indices
        self.align_corners = align_corners

    def forward(self, feats):
        low_level_feat = feats[self.backbone_indices[0]]
        shortcut = self.conv_bn_relu(low_level_feat)
        shortcut_shape = paddle.shape(shortcut)[2:]

        x = feats[self.backbone_indices[1]]
        x = self.depthwise_separable0(x)
        x = F.interpolate(
            x,
            shortcut_shape,
            mode='bilinear',
            align_corners=self.align_corners)
        x = paddle.concat(x=[shortcut, x], axis=1)
        x = self.depthwise_separable1(x)
        logit = self.depthwise_separable2(x)
        return [logit]


class _ConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 **kwargs):
        super().__init__()
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.KaimingUniform())
        self._conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=False,
            **kwargs)

        self._batch_norm = layers.SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x


class _ConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 **kwargs):
        super().__init__()
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.KaimingUniform())
        self._conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=False,
            **kwargs)

        self._batch_norm = layers.SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class _SeparableConvBNReLU(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise_conv = _ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size / 2),
            groups=in_channels,
            **kwargs)
        self.piontwise_conv = _ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=1,
            stride=1,
            padding=0)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


# if __name__ == '__main__':
#     import numpy as np
#     import os
#     import sys
#     sys.path.append(os.getcwd())
#     from backbones.lcnet import PPLCNet_x1_0

#     np.random.seed(100)
#     paddle.seed(100)

#     backbone = PPLCNet_x1_0()
#     net = PPHumanSegV1Lite(10, backbone)
#     img = np.random.random(size=(4, 3, 100, 100)).astype('float32')
#     img = paddle.to_tensor(img)
#     out = net(img)
#     print(out[0].shape)

#     # net.forward = paddle.jit.to_static(net.forward)
#     # save_path = os.path.join('.', 'model')
#     # in_var = paddle.ones([4, 3, 100, 100])
#     # paddle.jit.save(net, save_path, input_spec=[in_var])

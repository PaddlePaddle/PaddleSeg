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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class DANet(nn.Layer):
    """
    The DANet implementation based on PaddlePaddle.

    The original article refers to
    Fu, jun, et al. "Dual Attention Network for Scene Segmentation"
    (https://arxiv.org/pdf/1809.02983.pdf)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        backbone_indices (tuple): The values in the tuple indicate the indices of
            output of backbone.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices,
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        self.backbone_indices = backbone_indices
        in_channels = [self.backbone.feat_channels[i] for i in backbone_indices]

        self.head = DAHead(num_classes=num_classes, in_channels=in_channels)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feats = self.backbone(x)
        feats = [feats[i] for i in self.backbone_indices]
        logit_list = self.head(feats)
        if not self.training:
            logit_list = [logit_list[0]]

        logit_list = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners,
                align_mode=1) for logit in logit_list
        ]
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class DAHead(nn.Layer):
    """
    The Dual attention head.

    Args:
        num_classes (int): The unique number of target classes.
        in_channels (tuple): The number of input channels.
    """

    def __init__(self, num_classes, in_channels):
        super().__init__()
        in_channels = in_channels[-1]
        inter_channels = in_channels // 4

        self.channel_conv = layers.ConvBNReLU(in_channels, inter_channels, 3)
        self.position_conv = layers.ConvBNReLU(in_channels, inter_channels, 3)
        self.pam = PAM(inter_channels)
        self.cam = CAM(inter_channels)
        self.conv1 = layers.ConvBNReLU(inter_channels, inter_channels, 3)
        self.conv2 = layers.ConvBNReLU(inter_channels, inter_channels, 3)

        self.aux_head = nn.Sequential(
            nn.Dropout2D(0.1), nn.Conv2D(in_channels, num_classes, 1))

        self.aux_head_pam = nn.Sequential(
            nn.Dropout2D(0.1), nn.Conv2D(inter_channels, num_classes, 1))

        self.aux_head_cam = nn.Sequential(
            nn.Dropout2D(0.1), nn.Conv2D(inter_channels, num_classes, 1))

        self.cls_head = nn.Sequential(
            nn.Dropout2D(0.1), nn.Conv2D(inter_channels, num_classes, 1))

    def forward(self, feat_list):
        feats = feat_list[-1]
        channel_feats = self.channel_conv(feats)
        channel_feats = self.cam(channel_feats)
        channel_feats = self.conv1(channel_feats)

        position_feats = self.position_conv(feats)
        position_feats = self.pam(position_feats)
        position_feats = self.conv2(position_feats)

        feats_sum = position_feats + channel_feats
        logit = self.cls_head(feats_sum)

        if not self.training:
            return [logit]

        cam_logit = self.aux_head_cam(channel_feats)
        pam_logit = self.aux_head_cam(position_feats)
        aux_logit = self.aux_head(feats)
        return [logit, cam_logit, pam_logit, aux_logit]


class PAM(nn.Layer):
    """Position attention module."""

    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 8
        self.mid_channels = mid_channels
        self.in_channels = in_channels

        self.query_conv = nn.Conv2D(in_channels, mid_channels, 1, 1)
        self.key_conv = nn.Conv2D(in_channels, mid_channels, 1, 1)
        self.value_conv = nn.Conv2D(in_channels, in_channels, 1, 1)

        self.gamma = self.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0))

    def forward(self, x):
        x_shape = paddle.shape(x)

        # query: n, h * w, c1
        query = self.query_conv(x)
        query = paddle.reshape(query, (0, self.mid_channels, -1))
        query = paddle.transpose(query, (0, 2, 1))

        # key: n, c1, h * w
        key = self.key_conv(x)
        key = paddle.reshape(key, (0, self.mid_channels, -1))

        # sim: n, h * w, h * w
        sim = paddle.bmm(query, key)
        sim = F.softmax(sim, axis=-1)

        value = self.value_conv(x)
        value = paddle.reshape(value, (0, self.in_channels, -1))
        sim = paddle.transpose(sim, (0, 2, 1))

        # feat: from (n, c2, h * w) -> (n, c2, h, w)
        feat = paddle.bmm(value, sim)
        feat = paddle.reshape(feat,
                              (0, self.in_channels, x_shape[2], x_shape[3]))

        out = self.gamma * feat + x
        return out


class CAM(nn.Layer):
    """Channel attention module."""

    def __init__(self, channels):
        super().__init__()

        self.channels = channels
        self.gamma = self.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0))

    def forward(self, x):
        x_shape = paddle.shape(x)
        # query: n, c, h * w
        query = paddle.reshape(x, (0, self.channels, -1))
        # key: n, h * w, c
        key = paddle.reshape(x, (0, self.channels, -1))
        key = paddle.transpose(key, (0, 2, 1))

        # sim: n, c, c
        sim = paddle.bmm(query, key)
        # The danet author claims that this can avoid gradient divergence
        sim = paddle.max(sim, axis=-1, keepdim=True).tile(
            [1, 1, self.channels]) - sim
        sim = F.softmax(sim, axis=-1)

        # feat: from (n, c, h * w) to (n, c, h, w)
        value = paddle.reshape(x, (0, self.channels, -1))
        feat = paddle.bmm(sim, value)
        feat = paddle.reshape(feat, (0, self.channels, x_shape[2], x_shape[3]))

        out = self.gamma * feat + x
        return out

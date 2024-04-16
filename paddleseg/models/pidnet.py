# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import math
from typing import Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Constant, Normal, Uniform
from paddle import Tensor

from paddleseg.cvlibs import manager
from paddleseg.utils import utils


@manager.MODELS.add_component
class PIDNet(nn.Layer):
    """
    The PIDNet implementation based on PaddlePaddle.

    The original article refers to "Jiacong Xu, Zixiang Xiong, Shankar P. Bhattacharyya.
    PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller.
    https://arxiv.org/abs/2206.02066"


    Args:
        num_classes(int): The unique number of target classes.
        backbone(nn.Layer): Backbone network.
        head_channels(int): head channels.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 head_channels,
                 ignore_index=255,
                 pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.align_corners = True

        self.decode_head = PIDHead(in_channels=backbone.feat_channels[0],
                                   channels=head_channels,
                                   num_classes=num_classes)

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        feat = self.backbone(x)
        logit_list = self.decode_head(feat)
        return [
            F.interpolate(logit,
                          x.shape[2:],
                          mode='bilinear',
                          align_corners=self.align_corners)
            for logit in logit_list
        ]

    def loss_computation(self, logits_list, losses, data):
        label = paddle.cast(data['label'], 'int64')
        edge = paddle.cast(data['edge'], 'int64')
        bd_label = paddle.where(
            F.sigmoid(logits_list[2][:, 0, :, :]) > 0.8, label,
            self.ignore_index)

        loss_s = (
            losses['coef'][0] * losses['types'][0](logits_list[0], label) +
            losses['coef'][1] * losses['types'][1](logits_list[1], label))
        loss_b = losses['coef'][2] * losses['types'][2](logits_list[2], edge)
        loss_sb = losses['coef'][3] * losses['types'][3](logits_list[1],
                                                         bd_label)

        return [loss_s, loss_b, loss_sb]


class BasePIDHead(nn.Layer):
    """Base class for PID head.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.SyncBatchNorm(in_channels), nn.ReLU(),
            nn.Conv2D(in_channels,
                      channels,
                      kernel_size=3,
                      padding=1,
                      bias_attr=False))
        self.norm = nn.SyncBatchNorm(channels)
        self.act = nn.ReLU()

    def forward(self, x: Tensor, cls_seg: Optional[nn.Layer]) -> Tensor:
        """Forward function.
        Args:
            x (Tensor): Input tensor.
            cls_seg (nn.Layer, optional): The classification head.

        Returns:
            Tensor: Output tensor.
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if cls_seg is not None:
            x = cls_seg(x)
        return x


class PIDHead(nn.Layer):
    """Decode head for PIDNet.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of output channels.
        num_classes (int): Number of classes.
    """

    def __init__(self, in_channels: int, channels: int, num_classes: int):
        super().__init__()
        self.i_head = BasePIDHead(in_channels, channels)
        self.p_head = BasePIDHead(in_channels // 2, channels)
        self.d_head = BasePIDHead(in_channels // 2, in_channels // 4)
        self.p_cls_seg = nn.Conv2D(channels, num_classes, kernel_size=1)
        self.d_cls_seg = nn.Conv2D(in_channels // 4, 1, kernel_size=1)
        self.conv_seg = nn.Conv2D(channels, num_classes, kernel_size=1)

    def init_weights(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                fan_out = layer.weight.shape[0] * \
                          layer.weight.shape[2] * layer.weight.shape[3]
                std = math.sqrt(2) / math.sqrt(fan_out)
                Normal(0, std)(layer.weight)
                if layer.bias is not None:
                    fan_in = layer.weight.shape[1] * \
                             layer.weight.shape[2] * layer.weight.shape[3]
                    bound = 1 / math.sqrt(fan_in)
                    Uniform(-bound, bound)(layer.bias)
            elif isinstance(layer, (nn.BatchNorm2D, nn.SyncBatchNorm)):
                Constant(1)(layer.weight)
                Constant(0)(layer.bias)

    def forward(
            self, inputs: Union[Tensor,
                                Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.
        Args:
            inputs (Tensor | tuple[Tensor]): Input tensor or tuple of
                Tensor. When training, the input is a tuple of three tensors,
                (p_feat, i_feat, d_feat), and the output is a tuple of three
                tensors, (p_seg_logit, i_seg_logit, d_seg_logit).
                When inference, only the head of integral branch is used, and
                input is a tensor of integral feature map, and the output is
                the segmentation logit.

        Returns:
            Tensor | tuple[Tensor]: Output tensor or tuple of tensors.
        """
        if self.training:
            x_p, x_i, x_d = inputs
            x_p = self.p_head(x_p, self.p_cls_seg)
            x_i = self.i_head(x_i, self.conv_seg)
            x_d = self.d_head(x_d, self.d_cls_seg)
            return x_p, x_i, x_d
        else:
            x_i = self.i_head(inputs, self.conv_seg)
            return x_i,

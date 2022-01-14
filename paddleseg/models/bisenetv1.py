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

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class BiseNetV1(nn.Layer):
    """
    The BiSeNetV1 implementation based on PaddlePaddle.

    The original article refers to
    Yu, Changqian, et al. "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
    (https://paperswithcode.com/paper/bisenet-bilateral-segmentation-network-for)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone network, currently support Resnet18_vd/Resnet34_vd/Resnet50_vd/Resnet101_vd.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """
    def __init__(self,
                 num_classes,
                 backbone,
                 conv_channel=128,
                 pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.spatial_path = SpatialPath(3, 128)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            layers.ConvBNReLU(512, conv_channel, 1, bias_attr=False),
        )

        self.arms = nn.LayerList([
            AttentionRefinement(512, conv_channel),
            AttentionRefinement(256, conv_channel),
        ])
        self.refines = nn.LayerList([
            layers.ConvBNReLU(conv_channel,
                              conv_channel,
                              3,
                              stride=1,
                              padding=1,
                              bias_attr=False),
            layers.ConvBNReLU(conv_channel,
                              conv_channel,
                              3,
                              stride=1,
                              padding=1,
                              bias_attr=False),
        ])

        self.heads = nn.LayerList([
            BiSeNetHead(conv_channel, num_classes, 8, True),
            BiSeNetHead(conv_channel, num_classes, 8, True),
            BiSeNetHead(conv_channel * 2, num_classes, 8, False),
        ])

        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2, 1)

        self.pretrained = pretrained

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        spatial_out = self.spatial_path(x)
        context_blocks = self.backbone(x)
        context_blocks.reverse()

        global_context = self.global_context(context_blocks[0])
        global_context = F.interpolate(global_context,
                                       size=paddle.shape(context_blocks[0])[2:],
                                       mode='bilinear',
                                       align_corners=True)
        last_fm = global_context
        pred_out = []

        for i, (fm, arm, refine) in enumerate(
                zip(context_blocks[:2], self.arms, self.refines)):
            fm = arm(fm)
            fm += last_fm
            last_fm = F.interpolate(fm,
                                    size=paddle.shape(context_blocks[i +
                                                                     1])[2:],
                                    mode='bilinear',
                                    align_corners=True)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm

        concate_fm = self.ffm(spatial_out, context_out)
        pred_out.append(concate_fm)

        output = []
        if self.training:
            for i, head in enumerate(self.heads):
                out = head(pred_out[i])
                output.append(out)
        else:
            out = self.heads[-1](pred_out[-1])
            output.append(out)
        return output


class SpatialPath(nn.Layer):
    """
    SpatialPath module of BiseNetV1 model

    Args:
        in_channels (int): The number of input channels in spatial path module.
        out_channels (int): The number of output channels in spatial path module.
    """
    def __init__(self, in_channels, out_channels, inner_channel=64):
        super().__init__()
        self.conv_7x7 = layers.ConvBNReLU(in_channels,
                                          inner_channel,
                                          7,
                                          stride=2,
                                          padding=3,
                                          bias_attr=False)
        self.conv_3x3_1 = layers.ConvBNReLU(inner_channel,
                                            inner_channel,
                                            3,
                                            stride=2,
                                            padding=1,
                                            bias_attr=False)
        self.conv_3x3_2 = layers.ConvBNReLU(inner_channel,
                                            inner_channel,
                                            3,
                                            stride=2,
                                            padding=1,
                                            bias_attr=False)
        self.conv_1x1 = layers.ConvBNReLU(inner_channel,
                                          out_channels,
                                          1,
                                          bias_attr=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        x = self.conv_1x1(x)
        return x


class BiSeNetHead(nn.Layer):
    """
    BiSeNet head of BiseNetV1 model

    Args:
        in_channels (int): The number of input channels in spatial path module.
        out_channels (int): The number of output channels in spatial path module.
        scale (int, float): The scale factor of interpolation.
    """
    def __init__(self, in_channels, out_channels, scale, is_aux=False):
        super().__init__()
        inner_channel = 128 if is_aux else 64
        self.conv_3x3 = layers.ConvBNReLU(in_channels,
                                          inner_channel,
                                          3,
                                          stride=1,
                                          padding=1,
                                          bias_attr=False)
        self.conv_1x1 = nn.Conv2D(inner_channel, out_channels, 1)
        self.scale = scale

    def forward(self, x):
        x = self.conv_3x3(x)
        x = self.conv_1x1(x)
        if self.scale > 1:
            x = F.interpolate(x,
                              scale_factor=self.scale,
                              mode='bilinear',
                              align_corners=True)
        return x


class AttentionRefinement(nn.Layer):
    """
    AttentionRefinement module of BiseNetV1 model

    Args:
        in_channels (int): The number of input channels in spatial path module.
        out_channels (int): The number of output channels in spatial path module.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_3x3 = layers.ConvBNReLU(in_channels,
                                          out_channels,
                                          3,
                                          stride=1,
                                          padding=1,
                                          bias_attr=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            layers.ConvBNReLU(out_channels, out_channels, 1, bias_attr=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_3x3(x)
        se = self.channel_attention(x)
        x = x * se
        return x


class FeatureFusion(nn.Layer):
    """
    AttentionRefinement module of BiseNetV1 model

    Args:
        in_channels (int): The number of input channels in spatial path module.
        out_channels (int): The number of output channels in spatial path module.
        reduction (int): A factor shrinks convolutional channels. Default: 1.
    """
    def __init__(self, in_channels, out_channels, reduction=1):
        super().__init__()
        self.conv_1x1 = layers.ConvBNReLU(in_channels,
                                          out_channels,
                                          1,
                                          bias_attr=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            layers.ConvBNReLU(out_channels,
                              out_channels // reduction,
                              1,
                              bias_attr=False),
            layers.ConvBNReLU(out_channels // reduction,
                              out_channels,
                              1,
                              bias_attr=False),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        fm = paddle.concat([x1, x2], axis=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se
        return output

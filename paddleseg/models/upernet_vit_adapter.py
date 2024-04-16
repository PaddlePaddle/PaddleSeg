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
class UPerNetViTAdapter(nn.Layer):
    """
    The UPerNetViTAdapter implementation based on PaddlePaddle.

    The original article refers to
    Chen, Zhe, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. 
    "Vision Transformer Adapter for Dense Predictions." 
    (https://arxiv.org/abs/2205.08534).

    The implementation is based on https://github.com/czczup/ViT-Adapter

    Args:
        num_classes (int): The unique number of target classes.
        backbone (nn.Layer): The backbone network.
        backbone_indices (tuple | list): The values indicate the indices of output of backbone.
        channels (int, optional): The channels of inter layers in upernet head. Default: 512.
        pool_scales (list, optional): The scales in PPM. Default: [1, 2, 3, 6].
        dropout_ratio (float, optional): The dropout ratio for upernet head. Default: 0.1.
        aux_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        aux_channels (int, optional): The channels of inter layers in auxiliary head. Default: 256.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices,
                 channels=512,
                 pool_scales=[1, 2, 3, 6],
                 dropout_ratio=0.1,
                 aux_loss=True,
                 aux_channels=256,
                 align_corners=False,
                 pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        self.align_corners = align_corners

        in_channels = [self.backbone.feat_channels[i] for i in backbone_indices]
        self.head = UPerNetHead(num_classes=num_classes,
                                in_channels=in_channels,
                                channels=channels,
                                pool_scales=pool_scales,
                                dropout_ratio=dropout_ratio,
                                aux_loss=aux_loss,
                                aux_channels=aux_channels,
                                align_corners=align_corners)

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        feats = self.backbone(x)
        feats = [feats[i] for i in self.backbone_indices]
        logit_list = self.head(feats)
        logit_list = [
            F.interpolate(logit,
                          x.shape[2:],
                          mode='bilinear',
                          align_corners=self.align_corners)
            for logit in logit_list
        ]
        return logit_list


class ConvBNReLU(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias_attr=False,
                 **kwargs):
        super().__init__()
        self.conv = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size,
                              bias_attr=bias_attr,
                              **kwargs)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PPM(nn.Layer):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple | list): Pooling scales used in PPM.
        in_channels (int): Input channels.
        channels (int): Output Channels after modules, before conv_seg.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners):
        super().__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        self.stages = nn.LayerList()
        for pool_scale in pool_scales:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2D(output_size=(pool_scale, pool_scale)),
                    ConvBNReLU(in_channels=in_channels,
                               out_channels=channels,
                               kernel_size=1)))

    def forward(self, x):
        ppm_outs = []
        for ppm in self.stages:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(ppm_out,
                                              x.shape[2:],
                                              mode='bilinear',
                                              align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UPerNetHead(nn.Layer):
    """
    This head is the implementation of "Unified Perceptual Parsing for Scene Understanding".
    This is heavily based on https://github.com/czczup/ViT-Adapter

    Args:
        num_classes (int): The unique number of target classes.
        in_channels (list[int]): The channels of input features.
        channels (int, optional): The channels of inter layers in upernet head. Default: 512.
        pool_scales (list, optional): The scales in PPM. Default: [1, 2, 3, 6].
        dropout_ratio (float, optional): The dropout ratio for upernet head. Default: 0.1.
        aux_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        aux_channels (int, optional): The channels of inter layers in auxiliary head. Default: 256.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 channels,
                 pool_scales=[1, 2, 3, 6],
                 dropout_ratio=0.1,
                 aux_loss=False,
                 aux_channels=256,
                 align_corners=False):
        super().__init__()
        self.align_corners = align_corners

        # PSP Module
        self.psp_modules = PPM(pool_scales,
                               in_channels[-1],
                               channels,
                               align_corners=align_corners)
        self.bottleneck = ConvBNReLU(in_channels[-1] +
                                     len(pool_scales) * channels,
                                     channels,
                                     3,
                                     padding=1)
        # FPN Module
        self.lateral_convs = nn.LayerList()
        self.fpn_convs = nn.LayerList()
        for ch in in_channels[:-1]:  # skip the top layer
            l_conv = ConvBNReLU(ch, channels, 1)
            fpn_conv = ConvBNReLU(channels, channels, 3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvBNReLU(len(in_channels) * channels,
                                         channels,
                                         3,
                                         padding=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2D(dropout_ratio)
        else:
            self.dropout = None
        self.conv_seg = nn.Conv2D(channels, num_classes, kernel_size=1)

        self.aux_loss = aux_loss
        if self.aux_loss:
            self.aux_conv = ConvBNReLU(in_channels[2],
                                       aux_channels,
                                       3,
                                       padding=1)
            self.aux_conv_seg = nn.Conv2D(aux_channels,
                                          num_classes,
                                          kernel_size=1)

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = paddle.concat(psp_outs, axis=1)
        output = self.bottleneck(psp_outs)
        return output

    def forward(self, inputs):
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            upsampled = F.interpolate(laterals[i],
                                      laterals[i - 1].shape[2:],
                                      mode='bilinear',
                                      align_corners=self.align_corners)
            laterals[i - 1] = laterals[i - 1] + upsampled

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        fpn_outs.append(laterals[-1])  # append psp feature

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i],
                                        size=fpn_outs[0].shape[2:],
                                        mode='bilinear',
                                        align_corners=self.align_corners)
        fpn_outs = paddle.concat(fpn_outs, axis=1)
        output = self.fpn_bottleneck(fpn_outs)

        if self.dropout is not None:
            output = self.dropout(output)
        output = self.conv_seg(output)
        logits_list = [output]

        if self.aux_loss and self.training:
            aux_output = self.aux_conv(inputs[2])
            aux_output = self.aux_conv_seg(aux_output)
            logits_list.append(aux_output)

        return logits_list

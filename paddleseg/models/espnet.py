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

from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers


class PSPModule(nn.Layer):
    def __init__(self, in_channels, out_channels, sizes=4):
        super().__init__()
        self.stages = nn.LayerList([
            nn.Conv2D(in_channels,
                      in_channels,
                      kernel_size=3,
                      stride=1,
                      groups=in_channels,
                      padding='same',
                      bias_attr=False) for _ in range(sizes)
        ])
        self.project = layers.ConvBNPReLU(in_channels * (sizes + 1),
                                          out_channels,
                                          1,
                                          stride=1,
                                          bias_attr=False)

    def forward(self, feats):
        h, w = feats.shape[2], feats.shape[3]
        out = [feats]
        for stage in self.stages:
            feats = F.avg_pool2d(feats, kernel_size=3, stride=2, padding='same')
            upsampled = F.interpolate(stage(feats),
                                      size=(h, w),
                                      mode='bilinear',
                                      align_corners=True)
            out.append(upsampled)
        return self.project(paddle.concat(out, axis=1))


@manager.MODELS.add_component
class EESPNetV2(nn.Layer):
    """
    The ESPNetV2 implementation based on PaddlePaddle.

    The original article refers to
    Sachin Mehta, Mohammad Rastegari, Linda Shapiro, and Hannaneh Hajishirzi. "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network"
    (https://arxiv.org/abs/1811.11431).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone network, currently support ESPNet.
        drop_prob (float): The probability of dropout.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """
    def __init__(self, num_classes, backbone, drop_prob=0.1, pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.in_channels = self.backbone.out_channels
        self.proj_l4_c = layers.ConvBNPReLU(self.in_channels[3],
                                            self.in_channels[2],
                                            1,
                                            stride=1,
                                            bias_attr=False)
        psp_size = 2 * self.in_channels[2]
        self.eesp_psp = nn.Sequential(
            layers.EESP(psp_size,
                        psp_size // 2,
                        stride=1,
                        branches=4,
                        kernel_size_maximum=7),
            PSPModule(psp_size // 2, psp_size // 2),
        )

        self.project_l3 = nn.Sequential(
            nn.Dropout2D(p=drop_prob),
            nn.Conv2D(psp_size // 2, num_classes, 1, 1, bias_attr=False),
        )
        self.act_l3 = layers.BNPReLU(num_classes)
        self.project_l2 = layers.ConvBNPReLU(self.in_channels[1] + num_classes,
                                             num_classes,
                                             1,
                                             stride=1,
                                             bias_attr=False)
        self.project_l1 = nn.Sequential(
            nn.Dropout2D(p=drop_prob),
            nn.Conv2D(self.in_channels[0] + num_classes,
                      num_classes,
                      1,
                      1,
                      bias_attr=False),
        )

        self.pretrained = pretrained

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def hierarchicalUpsample(self, x, factor=3):
        for i in range(factor):
            x = F.interpolate(x,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=True)
        return x

    def forward(self, x):
        out_l1, out_l2, out_l3, out_l4 = self.backbone(x)

        out_l4_proj = self.proj_l4_c(out_l4)
        l4_to_l3 = F.interpolate(out_l4_proj,
                                 scale_factor=2,
                                 mode='bilinear',
                                 align_corners=True)
        merged_l3 = self.eesp_psp(paddle.concat([out_l3, l4_to_l3], axis=1))
        proj_merge_l3 = self.project_l3(merged_l3)
        proj_merge_l3 = self.act_l3(proj_merge_l3)

        l3_to_l2 = F.interpolate(proj_merge_l3,
                                 scale_factor=2,
                                 mode='bilinear',
                                 align_corners=True)
        merged_l2 = self.project_l2(paddle.concat([out_l2, l3_to_l2], axis=1))

        l2_to_l1 = F.interpolate(merged_l2,
                                 scale_factor=2,
                                 mode='bilinear',
                                 align_corners=True)
        merged_l1 = self.project_l1(paddle.concat([out_l1, l2_to_l1], axis=1))

        if self.training:
            return [
                F.interpolate(merged_l1,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=True),
                self.hierarchicalUpsample(proj_merge_l3),
            ]
        else:
            return [
                F.interpolate(merged_l1,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=True)
            ]

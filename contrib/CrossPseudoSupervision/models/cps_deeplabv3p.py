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

import paddle
import paddle.nn as nn

from cvlibs import manager
import models.initializer as init
from paddleseg.models.deeplab import DeepLabV3P

__all__ = ['CPSDeeplabV3P']


@manager.MODELS.add_component
class CPSDeeplabV3P(nn.Layer):
    """
    The CPS DeeplabV3P implementation based on PaddlePaddle.

    The original article refers to
    Xiaokang Chen, et, al. "Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision"
     (https://arxiv.org/abs/2106.01226)

    Args:
        num_classes (int): The unique number of target classes.
        backbone_l (paddle.nn.Layer): Backbone network for branch1, currently support Resnet50/Resnet101.
        backbone_r (paddle.nn.Layer): Backbone network for branch2, currently support Resnet50/Resnet101.
        bn_eps (float): CPS model's batchnorm initialization used.
        bn_monentum (float): CPS model's batchnorm initialization used.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
           Default: (0, 3).
        aspp_ratios (tuple, optional): The dilation rate using in ASSP module.
            If output_stride=16, aspp_ratios should be set as (1, 6, 12, 18).
            If output_stride=8, aspp_ratios is (1, 12, 24, 36).
            Default: (1, 6, 12, 18).
        aspp_out_channels (int, optional): The output channels of ASPP module. Default: 256.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: True.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
        data_format(str, optional): Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".
    """

    def __init__(self,
                 num_classes,
                 backbone_l,
                 backbone_r,
                 bn_eps,
                 bn_momentum,
                 backbone_indices=(0, 3),
                 aspp_ratios=(1, 6, 12, 18),
                 aspp_out_channels=256,
                 align_corners=True,
                 pretrained=None,
                 data_format="NCHW"):
        super().__init__()

        self.backbone1 = backbone_l
        self.backbone2 = backbone_r

        self.branch1 = DeepLabV3P(
            num_classes=num_classes,
            backbone=self.backbone1,
            backbone_indices=backbone_indices,
            aspp_ratios=aspp_ratios,
            aspp_out_channels=aspp_out_channels,
            align_corners=align_corners,
            pretrained=pretrained,
            data_format=data_format)

        self.branch2 = DeepLabV3P(
            num_classes=num_classes,
            backbone=self.backbone2,
            backbone_indices=backbone_indices,
            aspp_ratios=aspp_ratios,
            aspp_out_channels=aspp_out_channels,
            align_corners=align_corners,
            pretrained=pretrained,
            data_format=data_format)

        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.cps_init_weight()

    def forward(self, data):
        if not self.training:
            pred1 = self.branch1(data)
            return pred1

        return self.branch1(data), self.branch2(data)

    def cps_init_weight(self):
        norm_layer = nn.BatchNorm2D if paddle.distributed.ParallelEnv(
        ).nranks == 1 else nn.SyncBatchNorm
        # Initialize subnetwork one
        self.init_weight(
            self.branch1.head,
            init.kaiming_normal_,
            norm_layer,
            self.bn_eps,
            self.bn_momentum,
            mode='fan_in',
            nonlinearity='relu')

        # Initialize subnetwork two
        self.init_weight(
            self.branch2.head,
            init.kaiming_normal_,
            norm_layer,
            self.bn_eps,
            self.bn_momentum,
            mode='fan_in',
            nonlinearity='relu')

    def init_weight(self, module_list, conv_init, norm_layer, bn_eps,
                    bn_momentum, **kwargs):
        if isinstance(module_list, list):
            for feature in module_list:
                self.__init_weight(feature, conv_init, norm_layer, bn_eps,
                                   bn_momentum, **kwargs)
        else:
            self.__init_weight(module_list, conv_init, norm_layer, bn_eps,
                               bn_momentum, **kwargs)

    def __init_weight(self, feature, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs):
        for m in feature.sublayers():
            if isinstance(m, (nn.Conv1D, nn.Conv2D, nn.Conv3D)):
                conv_init(m.weight, **kwargs)
            elif isinstance(m, norm_layer):
                m._epsilon = bn_eps
                m._momentum = bn_momentum
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

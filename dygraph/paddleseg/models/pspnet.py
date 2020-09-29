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

import os

import paddle.nn.functional as F
from paddle import nn
from paddleseg.cvlibs import manager
from paddleseg.models.common import pyramid_pool
from paddleseg.models.common.layer_libs import ConvBNReLU, AuxLayer
from paddleseg.utils import utils


@manager.MODELS.add_component
class PSPNet(nn.Layer):
    """
    The PSPNet implementation based on PaddlePaddle.

    The original article refers to
        Zhao, Hengshuang, et al. "Pyramid scene parsing network."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
        (https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf)

    Args:
        num_classes (int): the unique number of target classes.
        backbone (Paddle.nn.Layer): backbone network, currently support Resnet50/101.
        model_pretrained (str): the path of pretrained model. Default to None.
        backbone_indices (tuple): two values in the tuple indicate the indices of output of backbone.
        pp_out_channels (int): output channels after Pyramid Pooling Module. Default to 1024.
        bin_sizes (tuple): the out size of pooled feature maps. Default to (1,2,3,6).
        enable_auxiliary_loss (bool): a bool values indicates whether adding auxiliary loss. Default to True.
        pretrained (str): the path of pretrained model. Default to None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(2, 3),
                 pp_out_channels=1024,
                 bin_sizes=(1, 2, 3, 6),
                 enable_auxiliary_loss=True,
                 pretrained=None):

        super(PSPNet, self).__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = PSPNetHead(
            num_classes, 
            backbone_indices,
            backbone_channels,
            pp_out_channels,
            bin_sizes,
            enable_auxiliary_loss)

        utils.load_entire_model(self, pretrained)

    def forward(self, input):
        feat_list = self.backbone(input)
        logit_list = self.head(feat_list)
        return [
            F.resize_bilinear(logit, input.shape[2:]) for logit in logit_list
        ]


class PSPNetHead(nn.Layer):
    """
    The PSPNetHead implementation.

    Args:
        num_classes (int): the unique number of target classes.
        backbone_indices (tuple): two values in the tuple indicate the indices of output of backbone.
            the first index will be taken as a deep-supervision feature in auxiliary layer;
            the second one will be taken as input of Pyramid Pooling Module (PPModule).
            Usually backbone consists of four downsampling stage, and return an output of
            each stage, so we set default (2, 3), which means taking feature map of the third
            stage (res4b22) in backbone, and feature map of the fourth stage (res5c) as input of PPModule.
        backbone_channels (tuple): the same length with "backbone_indices". It indicates the channels of corresponding index.
        pp_out_channels (int): output channels after Pyramid Pooling Module. Default to 1024.
        bin_sizes (tuple): the out size of pooled feature maps. Default to (1,2,3,6).
        enable_auxiliary_loss (bool): a bool values indicates whether adding auxiliary loss. Default to True.
    """

    def __init__(self,
                 num_classes,
                 backbone_indices=(2, 3),
                 backbone_channels=(1024, 2048),
                 pp_out_channels=1024,
                 bin_sizes=(1, 2, 3, 6),
                 enable_auxiliary_loss=True):

        super(PSPNetHead, self).__init__()

        self.backbone_indices = backbone_indices

        self.psp_module = pyramid_pool.PPModule(
            in_channels=backbone_channels[1],
            out_channels=pp_out_channels,
            bin_sizes=bin_sizes)

        self.conv = nn.Conv2d(
            in_channels=pp_out_channels,
            out_channels=num_classes,
            kernel_size=1)

        if enable_auxiliary_loss:

            self.auxlayer = AuxLayer(
                in_channels=backbone_channels[0],
                inter_channels=backbone_channels[0] // 4,
                out_channels=num_classes)

        self.enable_auxiliary_loss = enable_auxiliary_loss

        self.init_weight()

    def forward(self, feat_list):
        logit_list = []
        x = feat_list[self.backbone_indices[1]]
        x = self.psp_module(x)
        x = F.dropout(x, p=0.1)  # dropout_prob
        logit = self.conv(x)
        logit_list.append(logit)

        if self.enable_auxiliary_loss:
            auxiliary_feat = feat_list[self.backbone_indices[0]]
            auxiliary_logit = self.auxlayer(auxiliary_feat)
            logit_list.append(auxiliary_logit)

        return logit_list

    def init_weight(self, pretrained_model=None):
        """
        Initialize the parameters of model parts.
        """
        pass


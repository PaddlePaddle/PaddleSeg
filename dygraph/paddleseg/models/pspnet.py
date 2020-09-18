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
from paddleseg.models.common import layer_libs, pyramid_pool
from paddleseg.utils import utils


@manager.MODELS.add_component
class PSPNet(nn.Layer):
    """
    The PSPNet implementation based on PaddlePaddle.

    The orginal artile refers to
        Zhao, Hengshuang, et al. "Pyramid scene parsing network."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
        (https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf)

    Args:
        num_classes (int): the unique number of target classes.

        backbone (Paddle.nn.Layer): backbone network, currently support Resnet50/101.

        model_pretrained (str): the path of pretrained model. Defaullt to None.

        backbone_indices (tuple): two values in the tuple indicte the indices of output of backbone.
                        the first index will be taken as a deep-supervision feature in auxiliary layer;
                        the second one will be taken as input of Pyramid Pooling Module (PPModule).
                        Usually backbone consists of four downsampling stage, and return an output of
                        each stage, so we set default (2, 3), which means taking feature map of the third
                        stage (res4b22) in backbone, and feature map of the fourth stage (res5c) as input of PPModule.

        backbone_channels (tuple): the same length with "backbone_indices". It indicates the channels of corresponding index.

        pp_out_channels (int): output channels after Pyramid Pooling Module. Default to 1024.

        bin_sizes (tuple): the out size of pooled feature maps. Default to (1,2,3,6).

        enable_auxiliary_loss (bool): a bool values indictes whether adding auxiliary loss. Default to True.

    """

    def __init__(self,
                 num_classes,
                 backbone,
                 model_pretrained=None,
                 backbone_indices=(2, 3),
                 backbone_channels=(1024, 2048),
                 pp_out_channels=1024,
                 bin_sizes=(1, 2, 3, 6),
                 enable_auxiliary_loss=True):

        super(PSPNet, self).__init__()

        self.backbone = backbone
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
            
            self.auxlayer = layer_libs.AuxLayer(
                in_channels=backbone_channels[0], 
                inter_channels=backbone_channels[0] // 4,
                out_channels=num_classes)

        self.enable_auxiliary_loss = enable_auxiliary_loss

        self.init_weight(model_pretrained)

    def forward(self, input, label=None):

        logit_list = []
        _, feat_list = self.backbone(input)

        x = feat_list[self.backbone_indices[1]]
        x = self.psp_module(x)
        x = F.dropout(x, p=0.1)  # dropout_prob
        logit = self.conv(x)
        logit = F.resize_bilinear(logit, input.shape[2:])
        logit_list.append(logit)

        if self.enable_auxiliary_loss:
            auxiliary_feat = feat_list[self.backbone_indices[0]]
            auxiliary_logit = self.auxlayer(auxiliary_feat)
            auxiliary_logit = F.resize_bilinear(auxiliary_logit,
                                                input.shape[2:])
            logit_list.append(auxiliary_logit)

        return logit_list

    def init_weight(self, pretrained_model=None):
        """
        Initialize the parameters of model parts.
        Args:
            pretrained_model ([str], optional): the path of pretrained model. Defaults to None.
        """
        if pretrained_model is not None:
            if os.path.exists(pretrained_model):
                utils.load_pretrained_model(self, pretrained_model)
            else:
                raise Exception('Pretrained model is not found: {}'.format(
                    pretrained_model))

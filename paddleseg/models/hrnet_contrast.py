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


class ProjectionHead(nn.Layer):
    """
    The projection head used by contrast learning.
    Args:
        dim_in (int): The dimensions of input features.
        proj_dim (int): The output dimensions of projection head. Default: 256.
        proj (str): The type of projection head, only support 'linear' and 'convmlp'. Default: 'convmlp'
    """
    def __init__(self, dim_in, proj_dim=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2D(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2D(dim_in, dim_in, kernel_size=1),
                layers.SyncBatchNorm(dim_in),
                nn.ReLU(),
                nn.Conv2D(dim_in, proj_dim, kernel_size=1),
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, axis=1)


@manager.MODELS.add_component
class HRNetW48Contrast(nn.Layer):
    """
    The HRNetW48Contrast implementation based on PaddlePaddle.

    The original article refers to
    Wenguan Wang, Tianfei Zhou, et al. "Exploring Cross-Image Pixel Contrast for Semantic Segmentation"
    (https://arxiv.org/abs/2101.11939).

    Args:
        in_channels (int): The output dimensions of backbone.
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network, currently support HRNet_W48.
        drop_prob (float): The probability of dropout.
        proj_dim (int): The projection dimensions.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
    """
    def __init__(self, in_channels, num_classes, backbone, drop_prob, proj_dim, align_corners=False, pretrained=None):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = backbone
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.align_corners = align_corners
        self.pretrained = pretrained

        self.cls_head = nn.Sequential(
            nn.Conv2D(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            layers.SyncBatchNorm(in_channels),
            nn.ReLU(),
            nn.Dropout2D(drop_prob),
            nn.Conv2D(in_channels, num_classes, kernel_size=1, stride=1, bias_attr=False),
        )
        self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=self.proj_dim)
        
    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        feats = self.backbone(x)[0]
        out = self.cls_head(feats)
        if self.training:
            emb = self.proj_head(feats)
            return [F.interpolate(out, x.shape[2:], mode='bilinear', align_corners=self.align_corners), {'seg': out, 'embed': emb}]
        else:
            return [F.interpolate(out, x.shape[2:], mode='bilinear', align_corners=self.align_corners)]

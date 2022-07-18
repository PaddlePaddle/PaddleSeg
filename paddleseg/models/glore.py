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

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class GloRe(nn.Layer):
    """
    The GloRe implementation based on PaddlePaddle.

    The original article refers to:
       Chen, Yunpeng, et al. "Graph-Based Global Reasoning Networks"
       (https://arxiv.org/pdf/1811.12814.pdf)
    
    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network, currently support Resnet50/101.
        backbone_indices (tuple, optional): Two values in the tuple indicate the indices of output of backbone.
        gru_channels (int, optional): The number of input channels in GloRe Unit. Default: 512.
        gru_num_state (int, optional): The number of states in GloRe Unit. Default: 128.
        gru_num_node (tuple, optional): The number of nodes in GloRe Unit. Default: Default: 128.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(2, 3),
                 gru_channels=512,
                 gru_num_state=128,
                 gru_num_node=64,
                 enable_auxiliary_loss=True,
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.head = GloReHead(num_classes, backbone_indices, backbone_channels,
                              gru_channels, gru_num_state, gru_num_node,
                              enable_auxiliary_loss)
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        feat_list = self.backbone(x)
        logit_list = self.head(feat_list)
        return [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class GloReHead(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone_indices,
                 backbone_channels,
                 gru_channels=512,
                 gru_num_state=128,
                 gru_num_node=64,
                 enable_auxiliary_loss=True):
        super().__init__()

        in_channels = backbone_channels[1]
        self.conv_bn_relu = layers.ConvBNReLU(
            in_channels, gru_channels, 1, bias_attr=False)
        self.gru_module = GruModule(
            num_input=gru_channels,
            num_state=gru_num_state,
            num_node=gru_num_node)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2D(512, num_classes, kernel_size=1)
        self.auxlayer = layers.AuxLayer(
            in_channels=backbone_channels[0],
            inter_channels=backbone_channels[0] // 4,
            out_channels=num_classes)

        self.backbone_indices = backbone_indices
        self.enable_auxiliary_loss = enable_auxiliary_loss

    def forward(self, feat_list):

        logit_list = []
        x = feat_list[self.backbone_indices[1]]

        feature = self.conv_bn_relu(x)
        gru_output = self.gru_module(feature)
        output = self.dropout(gru_output)
        logit = self.classifier(output)
        logit_list.append(logit)

        if self.enable_auxiliary_loss:
            low_level_feat = feat_list[self.backbone_indices[0]]
            auxiliary_logit = self.auxlayer(low_level_feat)
            logit_list.append(auxiliary_logit)

        return logit_list


class GCN(nn.Layer):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1D(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1D(
            num_state, num_state, kernel_size=1, bias_attr=bias)

    def forward(self, x):
        h = self.conv1(paddle.transpose(x, perm=(0, 2, 1)))
        h = paddle.transpose(h, perm=(0, 2, 1))
        h = h + x
        h = self.relu(self.conv2(h))
        return h


class GruModule(nn.Layer):
    def __init__(self,
                 num_input=512,
                 num_state=128,
                 num_node=64,
                 normalize=False):
        super(GruModule, self).__init__()
        self.normalize = normalize
        self.num_state = num_state
        self.num_node = num_node
        self.reduction_dim = nn.Conv2D(num_input, num_state, kernel_size=1)
        self.projection_mat = nn.Conv2D(num_input, num_node, kernel_size=1)
        self.gcn = GCN(num_state=self.num_state, num_node=self.num_node)
        self.extend_dim = nn.Conv2D(
            self.num_state, num_input, kernel_size=1, bias_attr=False)
        self.extend_bn = layers.SyncBatchNorm(num_input, epsilon=1e-4)

    def forward(self, input):
        n, c, h, w = input.shape
        # B, C, H, W
        reduction_dim = self.reduction_dim(input)
        # B, N, H, W
        mat_B = self.projection_mat(input)
        # B, C, H*W
        reshaped_reduction = paddle.reshape(
            reduction_dim, shape=[n, self.num_state, h * w])
        # B, N, H*W
        reshaped_B = paddle.reshape(mat_B, shape=[n, self.num_node, h * w])
        # B, N, H*W
        reproject = reshaped_B
        # B, C, N
        node_state_V = paddle.matmul(
            reshaped_reduction, paddle.transpose(
                reshaped_B, perm=[0, 2, 1]))

        if self.normalize:
            node_state_V = node_state_V * (1. / reshaped_reduction.shape[2])

        # B, C, N
        gcn_out = self.gcn(node_state_V)
        # B, C, H*W
        Y = paddle.matmul(gcn_out, reproject)
        # B, C, H, W
        Y = paddle.reshape(Y, shape=[n, self.num_state, h, w])
        Y_extend = self.extend_dim(Y)
        Y_extend = self.extend_bn(Y_extend)

        out = input + Y_extend
        return out

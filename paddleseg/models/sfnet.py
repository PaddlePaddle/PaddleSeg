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

from paddleseg.models import layers
from paddleseg.cvlibs import manager
from paddleseg.utils import utils


@manager.MODELS.add_component
class SFNet(nn.Layer):
    """
    The SFNet implementation based on PaddlePaddle.

    The original article refers to
    Li, Xiangtai, et al. "Semantic Flow for Fast and Accurate Scene Parsing"
    (https://arxiv.org/pdf/2002.10120.pdf).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network, currently support Resnet50/101.
        backbone_indices (tuple): Four values in the tuple indicate the indices of output of backbone.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: False.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the feature size is even,
            e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices,
                 enable_auxiliary_loss=False,
                 align_corners=False,
                 pretrained=None):
        super(SFNet, self).__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        self.in_channels = [
            self.backbone.feat_channels[i] for i in backbone_indices
        ]
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.enable_auxiliary_loss = enable_auxiliary_loss
        if self.backbone.layers == 18:
            fpn_dim = 128
            inplane_head = 512
            fpn_inplanes = [64, 128, 256, 512]
        else:
            fpn_dim = 256
            inplane_head = 2048
            fpn_inplanes = [256, 512, 1024, 2048]

        self.head = SFNetHead(
            inplane=inplane_head,
            num_class=num_classes,
            fpn_inplanes=fpn_inplanes,
            fpn_dim=fpn_dim,
            enable_auxiliary_loss=self.enable_auxiliary_loss)
        self.init_weight()

    def forward(self, x):
        feats = self.backbone(x)
        feats = [feats[i] for i in self.backbone_indices]
        logit_list = self.head(feats)
        logit_list = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class SFNetHead(nn.Layer):
    """
    The SFNetHead implementation.

    Args:
        inplane (int): Input channels of PPM module.
        num_class (int): The unique number of target classes.
        fpn_inplanes (list): The feature channels from backbone.
        fpn_dim (int, optional): The input channels of FAM module. Default: 256.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: False.
    """

    def __init__(self,
                 inplane,
                 num_class,
                 fpn_inplanes,
                 fpn_dim=256,
                 enable_auxiliary_loss=False):
        super(SFNetHead, self).__init__()
        self.ppm = layers.PPModule(
            in_channels=inplane,
            out_channels=fpn_dim,
            bin_sizes=(1, 2, 3, 6),
            dim_reduction=True,
            align_corners=True)
        self.enable_auxiliary_loss = enable_auxiliary_loss
        self.fpn_in = []

        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2D(fpn_inplane, fpn_dim, 1),
                    layers.SyncBatchNorm(fpn_dim), nn.ReLU()))

        self.fpn_in = nn.LayerList(self.fpn_in)
        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(
                nn.Sequential(
                    layers.ConvBNReLU(fpn_dim, fpn_dim, 3, bias_attr=False)))
            self.fpn_out_align.append(
                AlignedModule(inplane=fpn_dim, outplane=fpn_dim // 2))
            if self.enable_auxiliary_loss:
                self.dsn.append(
                    nn.Sequential(layers.AuxLayer(fpn_dim, fpn_dim, num_class)))

        self.fpn_out = nn.LayerList(self.fpn_out)
        self.fpn_out_align = nn.LayerList(self.fpn_out_align)

        if self.enable_auxiliary_loss:
            self.dsn = nn.LayerList(self.dsn)

        self.conv_last = nn.Sequential(
            layers.ConvBNReLU(
                len(fpn_inplanes) * fpn_dim, fpn_dim, 3, bias_attr=False),
            nn.Conv2D(fpn_dim, num_class, kernel_size=1))

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.enable_auxiliary_loss:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()
        output_size = paddle.shape(fpn_feature_list[0])[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(
                F.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode='bilinear',
                    align_corners=True))
        fusion_out = paddle.concat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        if self.enable_auxiliary_loss:
            out.append(x)
            return out
        else:
            return [x]


class AlignedModule(nn.Layer):
    """
    The FAM module implementation.

    Args:
       inplane (int): Input channles of FAM module.
       outplane (int): Output channels of FAN module.
       kernel_size (int, optional): Kernel size of semantic flow convolution layer. Default: 3.
    """

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2D(inplane, outplane, 1, bias_attr=False)
        self.down_l = nn.Conv2D(inplane, outplane, 1, bias_attr=False)
        self.flow_make = nn.Conv2D(
            outplane * 2,
            2,
            kernel_size=kernel_size,
            padding=1,
            bias_attr=False)

    def flow_warp(self, input, flow, size):
        input_shape = paddle.shape(input)
        norm = size[::-1].reshape([1, 1, 1, -1])
        norm.stop_gradient = True
        h_grid = paddle.linspace(-1.0, 1.0, size[0]).reshape([-1, 1])
        h_grid = h_grid.tile([size[1]])
        w_grid = paddle.linspace(-1.0, 1.0, size[1]).reshape([-1, 1])
        w_grid = w_grid.tile([size[0]]).transpose([1, 0])
        grid = paddle.concat([w_grid.unsqueeze(2), h_grid.unsqueeze(2)], axis=2)
        grid.unsqueeze(0).tile([input_shape[0], 1, 1, 1])
        grid = grid + paddle.transpose(flow, (0, 2, 3, 1)) / norm

        output = F.grid_sample(input, grid)
        return output

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        size = paddle.shape(low_feature)[2:]
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(
            h_feature, size=size, mode='bilinear', align_corners=True)
        flow = self.flow_make(paddle.concat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)
        return h_feature

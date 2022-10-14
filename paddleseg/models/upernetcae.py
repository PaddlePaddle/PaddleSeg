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

from paddleseg.utils import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


@manager.MODELS.add_component
class UPerNetCAE(nn.Layer):
    """ 
    Unified Perceptual Parsing for Scene Understanding
    (https://arxiv.org/abs/1807.10221)
    The UPerNet implementation based on PaddlePaddle.
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
                 channels,
                 fpn_dim,
                 inplane_head,
                 fpn_inplanes,
                 enable_auxiliary_loss=True,
                 align_corners=True,
                 dropout_ratio=0.1,
                 pretrained=None):
        super(UPerNetCAE, self).__init__()
        self._init_fpn(embed_dim=768, patch_size=16)
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        self.align_corners = align_corners
        self.pretrained = pretrained
        self.enable_auxiliary_loss = enable_auxiliary_loss

        self.fpn_dim = fpn_dim
        self.inplane_head = inplane_head
        self.fpn_inplanes = fpn_inplanes

        self.decode_head = UPerNetHead(
            inplane=inplane_head,
            num_class=num_classes,
            fpn_inplanes=fpn_inplanes,
            dropout_ratio=dropout_ratio,
            channels=channels,
            fpn_dim=fpn_dim,
            enable_auxiliary_loss=self.enable_auxiliary_loss)
        self.init_weight()

    def _init_fpn(self, embed_dim=768, patch_size=16, out_with_norm=False):
        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.Conv2DTranspose(
                    embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(
                    embed_dim, momentum=0.1),
                nn.GELU(),
                nn.Conv2DTranspose(
                    embed_dim, embed_dim, kernel_size=2, stride=2), )

            self.fpn2 = nn.Sequential(
                nn.Conv2DTranspose(
                    embed_dim, embed_dim, kernel_size=2, stride=2), )

            self.fpn3 = Identity()

            self.fpn4 = nn.MaxPool2D(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.Conv2DTranspose(
                    embed_dim, embed_dim, kernel_size=2, stride=2), )

            self.fpn2 = Identity()

            self.fpn3 = nn.Sequential(nn.MaxPool2D(kernel_size=2, stride=2), )

            self.fpn4 = nn.Sequential(nn.MaxPool2D(kernel_size=4, stride=4), )

        if not out_with_norm:
            self.norm = Identity()
        else:
            self.norm = nn.LayerNorm(embed_dim, epsilon=1e-6)

    def forward(self, x):
        feats, feats_shape = self.backbone(x)  # [1, 1024, 768]
        B, _, Hp, Wp = feats_shape

        feats = [feats[i] for i in self.backbone_indices]

        for i, feat in enumerate(feats):
            feats[i] = paddle.reshape(
                paddle.transpose(
                    self.norm(feat), perm=[0, 2, 1]),
                shape=[B, -1, Hp, Wp])

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(feats)):
            feats[i] = ops[i](feats[i])

        logit_list = self.decode_head(feats)
        logit_list = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=False) for logit in logit_list  # all ok
        ]
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class UPerNetHead(nn.Layer):
    """
    The UPerNetHead implementation.

    Args:
        inplane (int): Input channels of PPM module.
        num_class (int): The unique number of target classes.
        fpn_inplanes (list): The feature channels from backbone.
        fpn_dim (int, optional): The input channels of FPN module. Default: 512.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: False.
    """

    def __init__(self,
                 inplane,
                 num_class,
                 fpn_inplanes,
                 channels,
                 dropout_ratio=0.1,
                 fpn_dim=512,
                 enable_auxiliary_loss=False):
        super(UPerNetHead, self).__init__()
        self.psp_modules = layers.PPModule(
            in_channels=inplane,
            out_channels=fpn_dim,
            bin_sizes=(1, 2, 3, 6),
            dim_reduction=False,
            align_corners=False)

        self.enable_auxiliary_loss = enable_auxiliary_loss
        self.lateral_convs = []
        self.fpn_convs = []

        for fpn_inplane in fpn_inplanes[:-1]:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2D(
                        fpn_inplane, fpn_dim, 1, bias_attr=False),
                    layers.SyncBatchNorm(fpn_dim),
                    nn.ReLU()))
            self.fpn_convs.append(
                nn.Sequential(
                    layers.ConvBNReLU(
                        fpn_dim, fpn_dim, 3, bias_attr=False)))

        self.lateral_convs = nn.LayerList(self.lateral_convs)
        self.fpn_convs = nn.LayerList(self.fpn_convs)

        if self.enable_auxiliary_loss:
            if dropout_ratio is not None:
                self.dsn = nn.Sequential(
                    layers.ConvBNReLU(
                        fpn_inplanes[2], 256, 3, padding=1, bias_attr=False),
                    nn.Dropout2D(dropout_ratio),
                    nn.Conv2D(
                        256, num_class, kernel_size=1))
            else:
                self.dsn = nn.Sequential(
                    layers.ConvBNReLU(
                        fpn_inplanes[2], 256, 3, padding=1, bias_attr=False),
                    nn.Conv2D(
                        256, num_class, kernel_size=1))

        if dropout_ratio is not None:
            self.dropout = nn.Dropout2D(dropout_ratio)
        else:
            self.dropout = None

        self.fpn_bottleneck = layers.ConvBNReLU(
            len(fpn_inplanes) * channels,
            channels,
            3,
            padding=1,
            bias_attr=False)
        #self.conv_last = nn.Sequential(
        #    layers.ConvBNReLU(
        #        len(fpn_inplanes) * fpn_dim, fpn_dim, 3, bias_attr=False),
        #    nn.Conv2D(fpn_dim, num_class, kernel_size=1))
        self.conv_seg = nn.Conv2D(channels, num_class, kernel_size=1)

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, conv_out):
        psp_out = self.psp_modules(conv_out[-1])
        f = psp_out
        fpn_feature_list = [psp_out]
        out = []

        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.lateral_convs[i](conv_x)
            prev_shape = paddle.shape(conv_x)[2:]
            f = conv_x + F.interpolate(
                f, prev_shape, mode='bilinear', align_corners=False)
            fpn_feature_list.append(self.fpn_convs[i](f))

        fpn_feature_list.reverse()
        output_size = fpn_feature_list[0].shape[2:]
        # resize multi-scales feature
        for index in range(len(conv_out) - 1, 0, -1):
            fpn_feature_list[index] = F.interpolate(
                fpn_feature_list[index],
                size=output_size,
                mode='bilinear',
                align_corners=False)
        fusion_out = paddle.concat(fpn_feature_list, 1)
        x = self.fpn_bottleneck(fusion_out)
        x = self.cls_seg(x)

        if self.enable_auxiliary_loss:
            dsn = self.dsn(conv_out[2])
            out.append(x)
            out.append(dsn)
            return out
        else:
            return [x]

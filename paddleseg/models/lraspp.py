# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.models import layers
from paddleseg.cvlibs import manager


@manager.MODELS.add_component
class LRASPP(nn.Layer):
    """
    Semantic segmentation model with a light R-ASPP head.
    
    The original article refers to
        Howard, Andrew, et al. "Searching for mobilenetv3."
        (https://arxiv.org/pdf/1909.11065.pdf)

    Args:
        num_classes (int): The number of target classes.
        backbone(nn.Layer): Backbone network, such as stdc1net and resnet18. The backbone must
            has feat_channels, of which the length is 5.
        backbone_indices (List(int), optional): The values indicate the indices of backbone output 
            used as the input of the LR-ASPP head.
            Default: [0, 1, 3].
        lraspp_head_inter_chs (List(int), optional): The intermediate channels of LR-ASPP head.
            Default: [32, 64].
        lraspp_head_out_ch (int, optional): The output channels of each ASPP branch in the LR-ASPP head.
            Default: 128
        resize_mode (str, optional): The resize mode for the upsampling operation in the LR-ASPP head.
            Default: bilinear.
        use_gap (bool, optional): If true, use global average pooling in the LR-ASPP head; otherwise, use
            a 49x49 kernel for average pooling.
            Default: True.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=[0, 1, 3],
                 lraspp_head_inter_chs=[32, 64],
                 lraspp_head_out_ch=128,
                 resize_mode='bilinear',
                 use_gap=True,
                 pretrained=None):
        super().__init__()

        # backbone
        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should has feat_channels."
        assert len(backbone.feat_channels) >= len(backbone_indices), \
            f"The length of input backbone_indices ({len(backbone_indices)}) should not be" \
            f"greater than the length of feat_channels ({len(backbone.feat_channels)})."
        assert len(backbone.feat_channels) > max(backbone_indices), \
            f"The max value ({max(backbone_indices)}) of backbone_indices should be " \
            f"less than the length of feat_channels ({len(backbone.feat_channels)})."
        self.backbone = backbone

        assert len(backbone_indices) >= 1, "The lenght of backbone_indices " \
            "should not be lesser than 1"

        # head
        assert len(backbone_indices) == len(
            lraspp_head_inter_chs
        ) + 1, "The length of backbone_indices should be 1 greater than lraspp_head_inter_chs."
        self.backbone_indices = backbone_indices

        self.lraspp_head = LRASPPHead(backbone_indices, backbone.feat_channels,
                                      lraspp_head_inter_chs, lraspp_head_out_ch,
                                      num_classes, resize_mode, use_gap)

        # pretrained
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]

        feats_backbone = self.backbone(x)
        assert len(feats_backbone) >= len(self.backbone_indices), \
            f"The nums of backbone feats ({len(feats_backbone)}) should be greater or " \
            f"equal than the nums of backbone_indices ({len(self.backbone_indices)})"

        y = self.lraspp_head(feats_backbone)
        y = F.interpolate(y, x_hw, mode='bilinear', align_corners=False)
        logit_list = [y]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class LRASPPHead(nn.Layer):
    def __init__(self,
                 indices,
                 in_chs,
                 mid_chs,
                 out_ch,
                 n_classes,
                 resize_mode,
                 use_gap,
                 align_corners=False):
        super().__init__()

        self.indices = indices[-2::-1]
        self.in_chs = [in_chs[i] for i in indices[::-1]]
        self.mid_chs = mid_chs[::-1]
        self.convs = nn.LayerList()
        self.conv_ups = nn.LayerList()
        for in_ch, mid_ch in zip(self.in_chs[1:], self.mid_chs):
            self.convs.append(
                nn.Conv2D(
                    in_ch, mid_ch, kernel_size=1, bias_attr=False))
            self.conv_ups.append(layers.ConvBNReLU(out_ch + mid_ch, out_ch, 1))
        self.conv_w = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=(49, 49), stride=(16, 20))
            if not use_gap else nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(
                self.in_chs[0], out_ch, 1, bias_attr=False),
            nn.Sigmoid())
        self.conv_v = layers.ConvBNReLU(self.in_chs[0], out_ch, 1)
        self.conv_t = nn.Conv2D(out_ch, out_ch, kernel_size=1, bias_attr=False)
        self.conv_out = nn.Conv2D(
            out_ch, n_classes, kernel_size=1, bias_attr=False)

        self.interp = partial(
            F.interpolate, mode=resize_mode, align_corners=align_corners)

    def forward(self, in_feat_list):
        x = in_feat_list[-1]

        x = self.conv_v(x) * self.interp(self.conv_w(x), paddle.shape(x)[2:])
        y = self.conv_t(x)

        for idx, conv, conv_up in zip(self.indices, self.convs, self.conv_ups):
            feat = in_feat_list[idx]
            y = self.interp(y, paddle.shape(feat)[2:])
            y = paddle.concat([y, conv(feat)], axis=1)
            y = conv_up(y)

        y = self.conv_out(y)
        return y

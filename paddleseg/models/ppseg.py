# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
from paddleseg.models import layers
from paddleseg.cvlibs import manager
from paddleseg.utils import utils


@manager.MODELS.add_component
class PPSeg(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices,
                 head_type='PPSegHead_1',
                 arm_type='FusionAdd',
                 cp_out_ch=128,
                 arm_out_chs=[128],
                 seg_head_mid_chs=[64],
                 resize_mode='nearest',
                 use_boundary_2=False,
                 use_boundary_4=False,
                 use_boundary_8=False,
                 use_boundary_16=False,
                 pretrained=None):
        super().__init__()

        print("backbone type: " + backbone.__class__.__name__)
        print("backbone_indices: " + str(backbone_indices))
        print("head_type: " + head_type)
        print("arm_type: " + arm_type)
        print("cp_out_ch: " + str(cp_out_ch))
        print("arm_out_chs: " + str(arm_out_chs))
        print("seg_head_mid_chs: " + str(seg_head_mid_chs))
        print("resize_mode: " + resize_mode)

        # backbone
        self.backbone = backbone
        backbone_name = backbone.__class__.__name__

        assert len(backbone_indices) > 1, "The lenght of backbone_indices " \
            "should be greater than 1"
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]

        # head
        head_class = eval(head_type)
        assert backbone_name in head_class.support_backbone, \
            "Not support backbone ({})".format(backbone_name)

        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
        assert len(arm_out_chs) == len(backbone_indices), "The length of " \
            "arm_out_chs and backbone_indices should be equal"

        self.ppseg_head = head_class(backbone_indices, backbone_out_chs,
                                     arm_out_chs, cp_out_ch, arm_type,
                                     resize_mode)

        if len(seg_head_mid_chs) == 1:
            seg_head_mid_chs = seg_head_mid_chs * len(backbone_indices)
        assert len(seg_head_mid_chs) == len(backbone_indices), "The length of " \
            "seg_head_mid_chs and backbone_indices should be equal"
        self.seg_heads = nn.LayerList()  # [..., head_16, head32]
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_mid_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, num_classes))

        # detail guidance
        mid_ch = 64
        if use_boundary_2 and len(backbone_out_chs) == 5:
            in_ch = backbone_out_chs[0]
            self.seg_head_sp2 = SegHead(in_ch, mid_ch, 1)
            print("Use boundary guidance of x2")
        if use_boundary_4 and len(backbone_out_chs) == 4:
            in_ch = backbone_out_chs[1]
            self.seg_head_sp4 = SegHead(in_ch, mid_ch, 1)
            print("Use boundary guidance of x4")
        if use_boundary_8 and len(backbone_out_chs) == 3:
            in_ch = backbone_out_chs[2]
            self.seg_head_sp8 = SegHead(in_ch, mid_ch, 1)
            print("Use boundary guidance of x8")
        if use_boundary_16 and len(backbone_out_chs) == 2:
            in_ch = backbone_out_chs[3]
            self.seg_head_sp16 = SegHead(in_ch, mid_ch, 1)
            print("Use boundary guidance of x16")

        # pretrained
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]

        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        assert len(feats_backbone) == 5

        feats_selected = [feats_backbone[i] for i in self.backbone_indices]
        feats_head = self.ppseg_head(feats_selected)  # [..., x16, x32]

        if self.training:
            logit_list = []

            for x, seg_head in zip(feats_head, self.seg_heads):
                x = seg_head(x)
                logit_list.append(x)

            logit_list = [
                F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
                for x in logit_list
            ]

            if hasattr(self, 'seg_head_sp2'):
                logit_list.append(self.seg_head_sp2(feats_backbone[0]))
            if hasattr(self, 'seg_head_sp4'):
                logit_list.append(self.seg_head_sp4(feats_backbone[1]))
            if hasattr(self, 'seg_head_sp8'):
                logit_list.append(self.seg_head_sp8(feats_backbone[2]))
            if hasattr(self, 'seg_head_sp16'):
                logit_list.append(self.seg_head_sp16(feats_backbone[3]))
        else:
            feat_out = self.seg_heads[0](feats_head[0])
            feat_out = F.interpolate(
                feat_out, x_hw, mode='bilinear', align_corners=False)
            logit_list = [feat_out]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class SegHead(nn.Layer):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            in_chan,
            mid_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.conv_out = nn.Conv2D(
            mid_chan, n_classes, kernel_size=1, bias_attr=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class PPSegHead_1(nn.Layer):
    '''
    The head of PPSeg.
    '''
    support_backbone = ["STDCNet_pp_1"]

    def __init__(self, backbone_indices, backbone_out_chs, arm_out_chs,
                 cp_out_ch, arm_type, resize_mode):
        super().__init__()

        assert len(backbone_indices) == len(backbone_out_chs), "The length of " \
            "backbone_indices and backbone_out_chs should be equal"
        assert len(backbone_indices) == len(arm_out_chs), "The length of " \
            "backbone_indices and arm_out_chs should be equal"

        self.cp = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            layers.ConvBNReLU(
                backbone_out_chs[-1], cp_out_ch, kernel_size=1,
                bias_attr=False))

        assert hasattr(layers,arm_type), \
            "Do not have arm_type ({})".format(arm_type)
        arm_class = eval("layers." + arm_type)

        self.arm_list = nn.LayerList()  # [..., arm8, arm16, arm32]
        for i in range(len(backbone_out_chs)):
            low_ch = backbone_out_chs[i]
            high_ch = cp_out_ch if i == len(
                backbone_out_chs) - 1 else arm_out_chs[i + 1]
            out_ch = arm_out_chs[i]
            arm = arm_class(
                low_ch, high_ch, out_ch, ksize=3, resize_mode=resize_mode)
            self.arm_list.append(arm)

        self.resize_mode = resize_mode

    def forward(self, in_feat_list):
        '''
        Args:
            in_feat_list (List(Tensor)): such as [x2, x4, x8, x16, x32]. x2, x4 and x8 are optional.
        Returns:
            out_feat_list (List(Tensor)): such as [x2, x4, x8, x16, x32]. x2, x4 and x8 are optional.
                The length of in_feat_list and out_feat_list are the same.
        '''

        high_feat = self.cp(in_feat_list[-1])
        out_feat_list = []

        for i in reversed(range(len(in_feat_list))):
            low_feat = in_feat_list[i]
            arm = self.arm_list[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)

        return out_feat_list

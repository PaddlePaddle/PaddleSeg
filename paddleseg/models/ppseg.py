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
                 backbone_indices=[2, 3, 4],
                 feat_nums=[1, 1, 1],
                 head_type='PPSegHead',
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
        print("feat_nums:" + str(feat_nums))
        print("head_type: " + head_type)
        print("arm_type: " + arm_type)
        print("cp_out_ch: " + str(cp_out_ch))
        print("arm_out_chs: " + str(arm_out_chs))
        print("seg_head_mid_chs: " + str(seg_head_mid_chs))
        print("resize_mode: " + resize_mode)

        # backbone
        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should has feat_channels."
        assert len(backbone.feat_channels) == 5, \
            "The length of feat_channels ({}) in backbone should be 5.".format(len(backbone.feat_channels))
        self.backbone = backbone

        assert len(backbone_indices) > 1, "The lenght of backbone_indices " \
            "should be greater than 1"
        assert len(backbone_indices) == len(feat_nums), "The length of " \
            "backbone_indices and feat_nums should be equal"
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        self.feat_nums = feat_nums
        backbone_out_chs = [[backbone.feat_channels[i]] * n \
            for i, n in zip(backbone_indices, feat_nums)]
        print("backbone_out_chs:" + str(backbone_out_chs))

        # head
        head_class = eval(head_type)

        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
        assert len(arm_out_chs) == len(backbone_indices), "The length of " \
            "arm_out_chs and backbone_indices should be equal"

        self.ppseg_head = head_class(backbone_indices, feat_nums,
                                     backbone_out_chs, arm_out_chs, cp_out_ch,
                                     arm_type, resize_mode)

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
            in_ch = backbone_out_chs[0][0]
            self.seg_head_sp2 = SegHead(in_ch, mid_ch, 1)
            print("Use boundary guidance of x2")
        if use_boundary_4 and len(backbone_out_chs) == 4:
            in_ch = backbone_out_chs[1][0]
            self.seg_head_sp4 = SegHead(in_ch, mid_ch, 1)
            print("Use boundary guidance of x4")
        if use_boundary_8 and len(backbone_out_chs) == 3:
            in_ch = backbone_out_chs[2][0]
            self.seg_head_sp8 = SegHead(in_ch, mid_ch, 1)
            print("Use boundary guidance of x8")
        if use_boundary_16 and len(backbone_out_chs) == 2:
            in_ch = backbone_out_chs[3][0]
            self.seg_head_sp16 = SegHead(in_ch, mid_ch, 1)
            print("Use boundary guidance of x16")

        # pretrained
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]

        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        assert len(
            feats_backbone) == 5, "The nums of backbone feats should be 5"

        tmp = []
        for fs in feats_backbone:
            fs = fs if isinstance(fs, (list, tuple)) else [fs]
            tmp.append(fs)
        feats_backbone = tmp
        '''
        feats_selected = [feats_backbone[i] for i in self.backbone_indices]
        for fs, num in zip(feats_selected, self.feat_nums):
            assert len(fs) == num, "The nums of feats is not equal to input num"
        '''

        feats_selected = []
        for idx, num in zip(self.backbone_indices, self.feat_nums):
            fs = feats_backbone[idx]
            assert len(fs) >= num, "The nums of backbone output feats should be " \
                "greater than the nums you set"
            fs = fs[-num:]
            feats_selected.append(fs)

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
                logit_list.append(self.seg_head_sp2(feats_backbone[0][-1]))
            if hasattr(self, 'seg_head_sp4'):
                logit_list.append(self.seg_head_sp4(feats_backbone[1][-1]))
            if hasattr(self, 'seg_head_sp8'):
                logit_list.append(self.seg_head_sp8(feats_backbone[2][-1]))
            if hasattr(self, 'seg_head_sp16'):
                logit_list.append(self.seg_head_sp16(feats_backbone[3][-1]))
        else:
            feat_out = self.seg_heads[0](feats_head[0])
            feat_out = F.interpolate(
                feat_out, x_hw, mode='bilinear', align_corners=False)
            logit_list = [feat_out]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class PPSegHead(nn.Layer):
    '''
    The head of PPSeg.
    '''

    def __init__(self, backbone_indices, feat_nums, backbone_out_chs,
                 arm_out_chs, cp_out_ch, arm_type, resize_mode):
        super().__init__()

        assert len(backbone_indices) == len(backbone_out_chs), "The length of " \
            "backbone_indices and backbone_out_chs should be equal"
        assert len(backbone_indices) == len(arm_out_chs), "The length of " \
            "backbone_indices and arm_out_chs should be equal"

        self.feat_nums = feat_nums
        self.resize_mode = resize_mode

        self.cp = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            layers.ConvBNReLU(
                backbone_out_chs[-1][-1],
                cp_out_ch,
                kernel_size=1,
                bias_attr=False))

        assert hasattr(layers,arm_type), \
            "Do not have arm_type ({})".format(arm_type)
        arm_class = eval("layers." + arm_type)

        self.arm_list = nn.LayerList()  # [..., arm8, arm16, arm32]
        for i in range(len(backbone_out_chs)):
            low_chs = backbone_out_chs[i]
            high_ch = cp_out_ch if i == len(
                backbone_out_chs) - 1 else arm_out_chs[i + 1]
            out_ch = arm_out_chs[i]
            arm = arm_class(
                low_chs, high_ch, out_ch, ksize=3, resize_mode=resize_mode)
            self.arm_list.append(arm)

    def forward(self, in_feat_list):
        '''
        Args:
            in_feat_list (List(Tensor)): such as [x2, x4, x8, x16, x32]. x2, x4 and x8 are optional.
        Returns:
            out_feat_list (List(Tensor)): such as [x2, x4, x8, x16, x32]. x2, x4 and x8 are optional.
                The length of in_feat_list and out_feat_list are the same.
        '''

        high_feat = self.cp(in_feat_list[-1][-1])
        out_feat_list = []

        for i in reversed(range(len(in_feat_list))):
            low_feat = in_feat_list[i]
            arm = self.arm_list[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)

        return out_feat_list


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

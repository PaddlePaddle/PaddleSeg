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
class PPLiteSeg(nn.Layer):
    """
    PPSeg.

    Args:
        num_classes (int): The unique number of target classes.
        backbone(nn.Layer): Backbone network, such as stdc1net and resnet18. The backbone must
            has feat_channels, of which the length is 5.
        backbone_indices (List(int), optional): The values indicate the indices of output of backbone.
            Default: [2, 3, 4].
        feat_nums (List(int), optinal):  The values indicate the num of tensor used for each fusion module.
            Default: [1, 1, 1].
        head_type (str, optional): The head type of PPSeg. Default: PPSegHead.
        arm_type (str, optional): The type of attention refinement module (ARM). Default: ARM_Add_Add.
        cm_out_ch (int, optional): The channel of the last context module, which comes after backbone.
            Default: 128.
        arm_out_chs (List(int), optional): The out channels of each arm module. Default: [64, 64, 64].
        pretrained (str, optional): The path or url of pretrained model. Default: None.

    """
    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=[2, 3, 4],
                 feat_nums=[1, 1, 1],
                 feat_select_mode='last',
                 head_type='PPSegHead',
                 arm_type='ARM_Add_SpAttenAdd3',
                 cm_bin_sizes=[1, 2, 4],
                 cm_out_ch=128,
                 arm_out_chs=[64, 96, 128],
                 seg_head_mid_chs=[64, 64, 64],
                 eval_seg_head_id=0,
                 resize_mode='nearest',
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

        assert len(backbone_indices) > 1, "The lenght of backbone_indices " \
            "should be greater than 1"
        assert len(backbone_indices) == len(feat_nums), "The length of " \
            "backbone_indices and feat_nums should be equal"
        assert all([x in (1, 2) for x in feat_nums]), "The values in feat_nums " \
            "should be 1 or 2"
        assert feat_select_mode in ('last', 'even'), "feat_select_mode " \
            "should in ('last', 'even')"
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        self.feat_nums = feat_nums
        self.feat_select_mode = feat_select_mode
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
                                     backbone_out_chs, arm_out_chs,
                                     cm_bin_sizes, cm_out_ch, arm_type,
                                     resize_mode)

        if len(seg_head_mid_chs) == 1:
            seg_head_mid_chs = seg_head_mid_chs * len(backbone_indices)
        assert len(seg_head_mid_chs) == len(backbone_indices), "The length of " \
            "seg_head_mid_chs and backbone_indices should be equal"
        self.seg_heads = nn.LayerList()  # [..., head_16, head32]
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_mid_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, num_classes))

        self.eval_seg_head_id = eval_seg_head_id

        # pretrained
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]

        feats_backbone = self.backbone(x)  # [..., x4, x8, x16, x32]
        assert len(feats_backbone) >= len(self.backbone_indices), \
            f"The nums of backbone feats ({len(feats_backbone)}) should be greater or " \
            f"equal than the nums of backbone_indices ({len(self.backbone_indices)})"

        tmp = []
        for fs in feats_backbone:
            fs = fs if isinstance(fs, (list, tuple)) else [fs]
            tmp.append(fs)
        feats_backbone = tmp

        feats_selected = []
        for idx, num in zip(self.backbone_indices, self.feat_nums):
            fs = feats_backbone[idx]
            assert len(fs) >= num, "The nums of backbone output feats should be " \
                "greater than the nums you set"
            tmp = [fs[-1]]
            if num == 2:
                if self.feat_select_mode == 'last':
                    tmp.insert(0, fs[-2])
                else:
                    tmp.insert(0, fs[(len(fs) - 1) // 2])
            feats_selected.append(tmp)

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
            idx = self.eval_seg_head_id
            feat_out = self.seg_heads[idx](feats_head[idx])
            feat_out = F.interpolate(feat_out,
                                     x_hw,
                                     mode='bilinear',
                                     align_corners=False)
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
                 arm_out_chs, cm_bin_sizes, cm_out_ch, arm_type, resize_mode):
        super().__init__()

        assert len(backbone_indices) == len(backbone_out_chs), "The length of " \
            "backbone_indices and backbone_out_chs should be equal"
        assert len(backbone_indices) == len(arm_out_chs), "The length of " \
            "backbone_indices and arm_out_chs should be equal"

        self.feat_nums = feat_nums
        self.resize_mode = resize_mode
        '''
        self.cm = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            layers.ConvBNReLU(
                backbone_out_chs[-1][-1],
                cm_out_ch,
                kernel_size=1,
                bias_attr=False))
        '''
        self.cm = PPContextModule(backbone_out_chs[-1][-1], cm_out_ch,
                                  cm_out_ch, cm_bin_sizes)

        assert hasattr(layers,arm_type), \
            "Do not have arm_type ({})".format(arm_type)
        arm_class = eval("layers." + arm_type)

        self.arm_list = nn.LayerList()  # [..., arm8, arm16, arm32]
        for i in range(len(backbone_out_chs)):
            low_chs = backbone_out_chs[i]
            high_ch = cm_out_ch if i == len(
                backbone_out_chs) - 1 else arm_out_chs[i + 1]
            out_ch = arm_out_chs[i]
            arm = arm_class(low_chs,
                            high_ch,
                            out_ch,
                            ksize=3,
                            resize_mode=resize_mode)
            self.arm_list.append(arm)

    def forward(self, in_feat_list):
        '''
        Args:
            in_feat_list (List(Tensor)): such as [x2, x4, x8, x16, x32]. x2, x4 and x8 are optional.
        Returns:
            out_feat_list (List(Tensor)): such as [x2, x4, x8, x16, x32]. x2, x4 and x8 are optional.
                The length of in_feat_list and out_feat_list are the same.
        '''

        high_feat = self.cm(in_feat_list[-1][-1])
        out_feat_list = []

        for i in reversed(range(len(in_feat_list))):
            low_feat = in_feat_list[i]
            arm = self.arm_list[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)

        return out_feat_list


class PPContextModule(nn.Layer):
    """
    Lite Context module.

    Args:
        in_channels (int): The number of input channels to pyramid pooling module.
        inter_channels (int): The number of inter channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=False):
        super().__init__()

        self.stages = nn.LayerList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_out = layers.ConvBNReLU(in_channels=inter_channels,
                                          out_channels=out_channels,
                                          kernel_size=3,
                                          padding=1)

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2D(output_size=size)
        conv = layers.ConvBNReLU(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1)
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = paddle.shape(input)[2:]

        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(x,
                              input_shape,
                              mode='bilinear',
                              align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out += x

        out = self.conv_out(out)
        return out


class SegHead(nn.Layer):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = layers.ConvBNReLU(in_chan,
                                      mid_chan,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias_attr=False)
        self.conv_out = nn.Conv2D(mid_chan,
                                  n_classes,
                                  kernel_size=1,
                                  bias_attr=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

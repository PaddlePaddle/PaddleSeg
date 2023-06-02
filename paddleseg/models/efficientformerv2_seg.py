# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np

import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager, param_init
from paddleseg.utils import utils


@manager.MODELS.add_component
class EfficientFormerSeg(nn.Layer):
    """
    The EfficientFormerV2 implementation based on PaddlePaddle.

    The original article refers to Yanyu Li, Ju Hu, Yang Wen, Georgios Evangelidis, 
    Kamyar Salahi, Yanzhi Wang, Sergey Tulyakov, Jian Ren. 
    "Rethinking Vision Transformers for MobileNet Size and Speed". 
    (https://arxiv.org/pdf/2212.08059.pdf).

    Args:
        backbone (paddle.nn.Layer): Backbone networks.
        num_classes (int): The unique number of target classes.
        backbone_indices (list[int], optional): The values in the tuple indicate the indices of output of backbone.
            Default: [0, 1, 2, 3].
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        head_channels (int, optional): The number of hidden channels of segmentation head. Default: 128.
        add_extra_convs (bool|str): An argument about whether to add extra conv or not or where to add extra conv. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 backbone,
                 num_classes,
                 backbone_indices=[0, 1, 2, 3],
                 align_corners=False,
                 head_channels=128,
                 add_extra_convs=False,
                 pretrained=None):
        super().__init__()
        self.align_corners = align_corners
        self.backbone = backbone
        self.pretrained = pretrained
        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]
        self.neck = EfficientFormerFPNNeck(
            in_channels=backbone_channels,
            out_channels=256,
            num_outs=4,
            add_extra_convs=add_extra_convs)
        self.head = EfficientFormerFPN(
            in_channels=[256, 256, 256, 256],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=head_channels,
            dropout_ratio=0.1,
            num_classes=num_classes,
            align_corners=self.align_corners)

        self.init_weight()

    def forward(self, x):
        H, W = x.shape[-2:]
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        x = [
            F.interpolate(
                x,
                size=[H, W],
                mode='bilinear',
                align_corners=self.align_corners)
        ]

        return x

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class EfficientFormerFPNNeck(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=False,
                 relu_before_extra_convs=False):
        super().__init__()

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.LayerList()
        self.fpn_convs = nn.LayerList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2D(in_channels[i], out_channels, 1)
            fpn_conv = nn.Conv2D(out_channels, out_channels, 3, padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = nn.Conv2D(
                    in_channels, out_channels, 3, stride=2, padding=1)
                self.fpn_convs.append(extra_fpn_conv)
        self.init_weight()

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='nearest',
                align_corners=False)

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

    def init_weight(self):
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                param_init.kaiming_normal_init(sublayer.weight)
            elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(sublayer.weight, value=1.0)
                param_init.constant_init(sublayer.bias, value=0.0)


class EfficientFormerFPN(nn.Layer):
    def __init__(self,
                 in_index=[0, 1, 2, 3],
                 in_channels=[256, 256, 256, 256],
                 channels=128,
                 feature_strides=[4, 8, 16, 32],
                 dropout_ratio=0.1,
                 num_classes=150,
                 align_corners=False):
        super().__init__()

        self.in_channels = in_channels

        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]

        self.in_index = in_index

        self.channels = channels
        self.feature_strides = feature_strides
        self.align_corners = align_corners
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2D(self.dropout_ratio)

        self.scale_heads = nn.LayerList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    nn.Sequential(
                        nn.Conv2D(
                            self.in_channels[i] if k == 0 else self.channels,
                            self.channels,
                            3,
                            padding=1,
                            bias_attr=False),
                        nn.BatchNorm2D(self.channels),
                        nn.ReLU()))

                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))

            self.scale_heads.append(nn.Sequential(*scale_head))
        self.cls_seg = nn.Conv2D(self.channels, self.num_classes, kernel_size=1)
        self.init_weight()

    def forward(self, inputs):

        x = [inputs[i] for i in self.in_index]

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + F.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        if self.dropout_ratio > 0:
            output = self.dropout(output)
        output = self.cls_seg(output)
        return output

    def init_weight(self):
        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                param_init.kaiming_normal_init(sublayer.weight)
            elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(sublayer.weight, value=1.0)
                param_init.constant_init(sublayer.bias, value=0.0)

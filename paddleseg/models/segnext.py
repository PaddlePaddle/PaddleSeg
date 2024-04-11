# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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
from paddleseg.models.backbones.mscan import MSCAN
from paddleseg.models.layers import NMF2D, ConvGNAct
from paddleseg.utils import utils


@manager.MODELS.add_component
class SegNeXt(nn.Layer):
    """
    The SegNeXt implementation based on PaddlePaddle.

    The original article refers to
    Guo, Meng-Hao, et al. "SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation"
    (https://arxiv.org/pdf/2209.08575.pdf)

    Args:
        backbone (nn.Layer): The backbone must be an instance of MSCAN.
        decoder_cfg (dict): The arguments of decoder.
        num_classes (int): The unique number of target classes.
        backbone_indices (list(int), optional): The values indicate the indices of backbone output 
           used as the input of the SegNeXt head. Default: [1, 2, 3].
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 backbone,
                 decoder_cfg,
                 num_classes,
                 backbone_indices=[1, 2, 3],
                 pretrained=None):
        super().__init__()
        self.backbone = backbone

        in_channels = [self.backbone.feat_channels[i] for i in backbone_indices]
        self.decode_head = LightHamHead(in_channels=in_channels,
                                        num_classes=num_classes,
                                        **decoder_cfg)

        self.align_corners = self.decode_head.align_corners
        self.pretrained = pretrained
        self.init_weights()

    def init_weights(self):
        if self.pretrained:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        input_size = x.shape[2:]
        feats = self.backbone(x)
        out = self.decode_head(feats)
        return [
            F.interpolate(out,
                          input_size,
                          mode="bilinear",
                          align_corners=self.align_corners)
        ]


class Hamburger(nn.Layer):

    def __init__(self, ham_channels=512, num_groups=32, ham_kwargs=None):
        super().__init__()
        self.ham_in = nn.Conv2D(ham_channels, ham_channels, kernel_size=1)

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvGNAct(ham_channels,
                                 ham_channels,
                                 kernel_size=1,
                                 num_groups=num_groups,
                                 bias_attr=False)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy)

        return ham


class LightHamHead(nn.Layer):
    """The head implementation of HamNet based on PaddlePaddle.
    The original article refers to Zhengyang Geng, et al. "Is Attention Better Than Matrix Decomposition?"
    (https://arxiv.org/abs/2109.04553.pdf)

    Args:
        in_channels (list[int]): The feature channels from backbone.
        num_classes (int): The unique number of target classes.
        channels (int, optional): The intermediate channel of LightHamHead. Default: 256.
        dropout_rate (float, optional): The rate of dropout. Default: 0.1.
        align_corners (bool, optional): Whether use align_corners when interpolating. Default: False.
        ham_channels (int, optional): Input channel of Hamburger. Default: 512.
        num_groups (int, optional): The num_groups of convolutions in LightHamHead. Default: 32.
        ham_kwargs (dict, optional): Keyword arguments of Hamburger module.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 channels=256,
                 dropout_rate=0.1,
                 align_corners=False,
                 ham_channels=512,
                 num_groups=32,
                 ham_kwargs=None):
        super().__init__()

        if len(in_channels) != 3:
            raise ValueError(
                "The length of `in_channels` must be 3, but got {}".format(
                    len(in_channels)))

        self.align_corners = align_corners

        self.squeeze = ConvGNAct(sum(in_channels),
                                 ham_channels,
                                 kernel_size=1,
                                 num_groups=num_groups,
                                 act_type="relu",
                                 bias_attr=False)

        self.hamburger = Hamburger(ham_channels, num_groups, ham_kwargs)

        self.align = ConvGNAct(ham_channels,
                               channels,
                               kernel_size=1,
                               num_groups=num_groups,
                               act_type="relu",
                               bias_attr=False)

        self.dropout = (nn.Dropout2D(dropout_rate)
                        if dropout_rate > 0.0 else nn.Identity())
        self.conv_seg = nn.Conv2D(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        inputs = inputs[1:]
        target_shape = inputs[0].shape[2:]
        inputs = [
            F.interpolate(level,
                          size=target_shape,
                          mode="bilinear",
                          align_corners=self.align_corners) for level in inputs
        ]

        inputs = paddle.concat(inputs, axis=1)
        x = self.squeeze(inputs)

        x = self.hamburger(x)

        output = self.align(x)
        output = self.dropout(output)
        output = self.conv_seg(output)
        return output

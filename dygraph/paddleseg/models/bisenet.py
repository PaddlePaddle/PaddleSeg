# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models.common.layer_libs import ConvBNReLU, ConvBN, DepthwiseConvBN
from paddleseg.models.common.activation import Activation


class StemBlock(nn.Layer):
    def __init__(self, in_dim, out_dim):
        super(StemBlock, self).__init__()

        self.conv = ConvBNReLU(in_dim, out_dim, 3, stride=2)

        self.left = nn.Sequential(
            ConvBNReLU(out_dim, out_dim // 2, 1),
            ConvBNReLU(out_dim // 2, out_dim, 3, stride=2))

        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fuse = ConvBNReLU(out_dim * 2, out_dim, 3)

    def forward(self, x):
        x = self.conv(x)
        left = self.left(x)
        right = self.right(x)
        concat = paddle.concat([left, right], axis=1)
        return self.fuse(concat)


class ContextEmbeddingBlock(nn.Layer):
    def __init__(self, in_dim, out_dim):
        super(ContextEmbeddingBlock, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.SyncBatchNorm(in_dim)

        self.conv_1x1 = ConvBNReLU(in_dim, out_dim, 1)
        self.conv_3x3 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

    def forward(self, x):
        gap = self.gap(x)
        bn = self.bn(gap)
        conv1 = self.conv_1x1(bn) + x
        return self.conv_3x3(conv1)


class GatherAndExpandsionLayer1(nn.Layer):
    """Gather And Expandsion Layer with stride 1"""

    def __init__(self, in_dim, out_dim, expand):
        super(GatherAndExpandsionLayer1, self).__init__()

        expand_dim = expand * in_dim

        self.conv = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3), DepthwiseConvBN(
                in_dim, expand_dim, 3), ConvBN(expand_dim, out_dim, 1))

    def forward(self, x):
        return F.relu(self.conv(x) + x)


class GatherAndExpandsionLayer2(nn.Layer):
    """Gather And Expandsion Layer with stride 2"""

    def __init__(self, in_dim, out_dim, expand):
        super(GatherAndExpandsionLayer2, self).__init__()

        expand_dim = expand * in_dim

        self.branch_1 = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3),
            DepthwiseConvBN(in_dim, expand_dim, 3, stride=2),
            DepthwiseConvBN(expand_dim, expand_dim, 3),
            ConvBN(expand_dim, out_dim, 1))

        self.branch_2 = nn.Sequential(
            DepthwiseConvBN(in_dim, in_dim, 3, stride=2),
            ConvBN(in_dim, out_dim, 1))

    def forward(self, x):
        return F.relu(self.branch_1(x) + self.branch_2(x))


class DetailBranch(nn.Layer):
    """The detail branch of BiSeNet, which has wide channels but shallow layers."""

    def __init__(self, in_channels):
        super(DetailBranch, self).__init__()

        C1, C2, C3 = in_channels

        self.convs = nn.Sequential(
            # stage 1
            ConvBNReLU(3, C1, 3, stride=2),
            ConvBNReLU(C1, C1, 3),
            # stage 2
            ConvBNReLU(C1, C2, 3, stride=2),
            ConvBNReLU(C2, C2, 3),
            ConvBNReLU(C2, C2, 3),
            # stage 3
            ConvBNReLU(C2, C3, 3, stride=2),
            ConvBNReLU(C3, C3, 3),
            ConvBNReLU(C3, C3, 3),
        )

    def forward(self, x):
        return self.convs(x)


class SemanticBranch(nn.Layer):
    """The semantic branch of BiSeNet, which has narrow channels but deep layers."""

    def __init__(self, in_channels):
        super(SemanticBranch, self).__init__()
        C1, C3, C4, C5 = in_channels

        self.stem = StemBlock(3, C1)

        self.stage3 = nn.Sequential(
            GatherAndExpandsionLayer2(C1, C3, 6),
            GatherAndExpandsionLayer1(C3, C3, 6))

        self.stage4 = nn.Sequential(
            GatherAndExpandsionLayer2(C3, C4, 6),
            GatherAndExpandsionLayer1(C4, C4, 6))

        self.stage5_4 = nn.Sequential(
            GatherAndExpandsionLayer2(C4, C5, 6),
            GatherAndExpandsionLayer1(C5, C5, 6),
            GatherAndExpandsionLayer1(C5, C5, 6),
            GatherAndExpandsionLayer1(C5, C5, 6))

        self.ce = ContextEmbeddingBlock(C5, C5)

    def forward(self, x):
        stage2 = self.stem(x)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5_4 = self.stage5_4(stage4)
        fm = self.ce(stage5_4)
        return stage2, stage3, stage4, stage5_4, fm


class BGA(nn.Layer):
    """The Bilateral Guided Aggregation Layer, used to fuse the semantic features and spatial features."""

    def __init__(self, out_dim):
        super(BGA, self).__init__()

        self.db_branch_keep = nn.Sequential(
            DepthwiseConvBN(out_dim, out_dim, 3), nn.Conv2d(
                out_dim, out_dim, 1))

        self.db_branch_down = nn.Sequential(
            ConvBN(out_dim, out_dim, 3, stride=2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        self.sb_branch_keep = nn.Sequential(
            DepthwiseConvBN(out_dim, out_dim, 3), nn.Conv2d(
                out_dim, out_dim, 1), Activation(act='sigmoid'))

        self.sb_branch_up = nn.Sequential(
            ConvBN(out_dim, out_dim, 3),
            nn.UpsamplingBilinear2d(scale_factor=4), Activation(act='sigmoid'))

        self.conv = ConvBN(out_dim, out_dim, 3)

    def forward(self, dfm, sfm):
        db_feat_keep = self.db_branch_keep(dfm)
        db_feat_down = self.db_branch_down(dfm)
        sb_feat_keep = self.sb_branch_keep(sfm)
        sb_feat_up = self.sb_branch_up(sfm)
        db_feat = db_feat_keep * sb_feat_up
        sb_feat = db_feat_down * sb_feat_keep
        sb_feat = F.resize_bilinear(sb_feat, db_feat.shape[2:])

        return self.conv(db_feat + sb_feat)


class SegHead(nn.Layer):
    def __init__(self, in_dim, mid_dim, num_classes):
        super(SegHead, self).__init__()

        self.conv_3x3 = nn.Sequential(
            ConvBNReLU(in_dim, mid_dim, 3), nn.Dropout(0.1))

        self.conv_1x1 = nn.Conv2d(mid_dim, num_classes, 1, 1)

    def forward(self, x):
        conv1 = self.conv_3x3(x)
        conv2 = self.conv_1x1(conv1)
        return conv2


@manager.MODELS.add_component
class BiSeNetV2(nn.Layer):
    """
    The BiSeNet V2 implementation based on PaddlePaddle.

    The original article refers to
        Yu, Changqian, et al. "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
        (https://arxiv.org/abs/2004.02147)

    Args:
        num_classes(int): the unique number of target classes.
        lambd(float): factor for controlling the size of semantic branch channels. Default to 0.25.
        pretrained(str): the path or url of pretrained model. Default to None.
    """

    def __init__(self, num_classes, lambd=0.25, pretrained=None):
        super(BiSeNetV2, self).__init__()

        C1, C2, C3 = 64, 64, 128
        db_channels = (C1, C2, C3)
        C1, C3, C4, C5 = int(C1 * lambd), int(C3 * lambd), 64, 128
        sb_channels = (C1, C3, C4, C5)
        mid_channels = 128

        self.db = DetailBranch(db_channels)
        self.sb = SemanticBranch(sb_channels)

        self.bga = BGA(mid_channels)
        self.aux_head1 = SegHead(C1, C1, num_classes)
        self.aux_head2 = SegHead(C3, C3, num_classes)
        self.aux_head3 = SegHead(C4, C4, num_classes)
        self.aux_head4 = SegHead(C5, C5, num_classes)
        self.head = SegHead(mid_channels, mid_channels, num_classes)

        self.init_weight(pretrained)

    def forward(self, x):
        dfm = self.db(x)
        feat1, feat2, feat3, feat4, sfm = self.sb(x)
        logit1 = self.aux_head1(feat1)
        logit2 = self.aux_head2(feat2)
        logit3 = self.aux_head3(feat3)
        logit4 = self.aux_head4(feat4)
        logit = self.head(self.bga(dfm, sfm))

        logits = [logit, logit1, logit2, logit3, logit4]
        logits = [F.resize_bilinear(logit, x.shape[2:]) for logit in logits]

        return logits

    def init_weight(self, pretrained=None):
        """
        Initialize the parameters of model parts.
        Args:
            pretrained ([str], optional): the path of pretrained model.. Defaults to None.
        """
        if pretrained is not None:
            if os.path.exists(pretrained):
                utils.load_entire_model(self, pretrained)
            else:
                raise Exception(
                    'Pretrained model is not found: {}'.format(pretrained))
        else:
            for sublayer in self.sublayers():
                if isinstance(sublayer, nn.Conv2d):
                    param_init.msra_init(sublayer.weight)
                elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                    param_init.constant_init(sublayer.weight, value=1.0)
                    param_init.constant_init(sublayer.bias, value=0.0)

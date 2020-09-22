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


class StemBlock(nn.Layer):
    def __init__(self, in_dim, out_dim):
        super(StemBlock, self).__init__()

        self.conv_3x3 = ConvBNReLU(in_dim, out_dim, 3, stride=2, padding=1)
        self.conv_1x1 = ConvBNReLU(out_dim, out_dim // 2, 1)
        self.conv2_3x3 = ConvBNReLU(
            out_dim // 2, out_dim, 3, stride=2, padding=1)
        self.conv3_3x3 = ConvBNReLU(out_dim * 2, out_dim, 3, padding=1)

        self.mpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        conv1 = self.conv_3x3(x)
        conv2 = self.conv_1x1(conv1)
        conv3 = self.conv2_3x3(conv2)
        pool = self.mpool(conv1)
        concat = paddle.concat([conv3, pool], axis=1)
        return self.conv3_3x3(concat)


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


class GatherAndExpandsionLayer(nn.Layer):
    def __init__(self, in_dim, out_dim, expand, stride):
        super(GatherAndExpandsionLayer, self).__init__()

        self.stride = stride
        self.conv_3x3 = ConvBNReLU(in_dim, out_dim, 3, padding=1)
        self.dwconv = DepthwiseConvBN(
            out_dim, expand * out_dim, 3, stride=stride, padding=1)
        self.dwconv2 = DepthwiseConvBN(
            expand * out_dim, expand * out_dim, 3, padding=1)
        self.dwconv3 = DepthwiseConvBN(
            in_dim, out_dim, 3, stride=stride, padding=1)
        self.conv_1x1 = ConvBN(expand * out_dim, out_dim, 1)
        self.conv2_1x1 = ConvBN(out_dim, out_dim, 1)

    def forward(self, x):
        conv1 = self.conv_3x3(x)
        fm = self.dwconv(conv1)
        residual = x
        if self.stride == 2:
            fm = self.dwconv2(fm)
            residual = self.dwconv3(residual)
            residual = self.conv2_1x1(residual)
        fm = self.conv_1x1(fm)
        return F.relu(fm + residual)


class DetailBranch(nn.Layer):
    """The detail branch of BiSeNet, which has wide channels but shallow layers."""

    def __init__(self, in_channels):
        super(DetailBranch, self).__init__()

        C1, C2, C3 = in_channels

        self.convs = nn.Sequential(
            # stage 1
            ConvBNReLU(3, C1, 3, stride=2, padding=1),
            ConvBNReLU(C1, C1, 3, padding=1),
            # stage 2
            ConvBNReLU(C1, C2, 3, stride=2, padding=1),
            ConvBNReLU(C2, C2, 3, padding=1),
            ConvBNReLU(C2, C2, 3, padding=1),
            # stage 3
            ConvBNReLU(C2, C3, 3, stride=2, padding=1),
            ConvBNReLU(C3, C3, 3, padding=1),
            ConvBNReLU(C3, C3, 3, padding=1),
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
            GatherAndExpandsionLayer(C1, C3, 6, 2),
            GatherAndExpandsionLayer(C3, C3, 6, 1))

        self.stage4 = nn.Sequential(
            GatherAndExpandsionLayer(C3, C4, 6, 2),
            GatherAndExpandsionLayer(C4, C4, 6, 1))

        self.stage5_4 = nn.Sequential(
            GatherAndExpandsionLayer(C4, C5, 6, 2),
            GatherAndExpandsionLayer(C5, C5, 6, 1),
            GatherAndExpandsionLayer(C5, C5, 6, 1),
            GatherAndExpandsionLayer(C5, C5, 6, 1))

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

        self.db_dwconv = DepthwiseConvBN(out_dim, out_dim, 3, padding=1)
        self.db_conv_1x1 = nn.Conv2d(out_dim, out_dim, 1, 1)
        self.db_conv_3x3 = ConvBN(out_dim, out_dim, 3, stride=2, padding=1)
        self.db_apool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.sb_conv_3x3 = ConvBN(out_dim, out_dim, 3, padding=1)
        self.sb_dwconv = DepthwiseConvBN(out_dim, out_dim, 3, padding=1)
        self.sb_conv_1x1 = nn.Conv2d(out_dim, out_dim, 1)

        self.conv = ConvBN(out_dim, out_dim, 3, padding=1)

    def forward(self, dfm, sfm):
        dconv1 = self.db_dwconv(dfm)
        dconv2 = self.db_conv_1x1(dconv1)
        dconv3 = self.db_conv_3x3(dfm)
        dpool = self.db_apool(dconv3)

        sconv1 = self.sb_conv_3x3(sfm)
        sconv1 = F.resize_bilinear(sconv1, dconv2.shape[2:])
        att1 = F.sigmoid(sconv1)
        sconv2 = self.sb_dwconv(sfm)
        att2 = self.sb_conv_1x1(sconv2)
        att2 = F.sigmoid(att2)

        fm = F.resize_bilinear(att2 * dpool, dconv2.shape[2:])
        _sum = att1 * dconv2 + fm
        return self.conv(_sum)


class SegHead(nn.Layer):
    def __init__(self, in_dim, out_dim, num_classes):
        super(SegHead, self).__init__()

        self.conv_3x3 = ConvBNReLU(in_dim, out_dim, 3)
        self.conv_1x1 = nn.Conv2d(out_dim, num_classes, 1, 1)

    def forward(self, x, label=None):
        conv1 = self.conv_3x3(x)
        conv2 = self.conv_1x1(conv1)
        pred = F.resize_bilinear(conv2, x.shape[2:])
        return pred


@manager.MODELS.add_component
class BiSeNet(nn.Layer):
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
        super(BiSeNet, self).__init__()

        C1, C2, C3, C4, C5 = 64, 64, 128, 64, 128
        db_channels = (C1, C2, C3)
        C1, C3 = int(C1 * lambd), int(C3 * lambd)
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

    def forward(self, x, label=None):
        dfm = self.db(x)
        feat1, feat2, feat3, feat4, sfm = self.sb(x)
        logit1 = self.aux_head1(feat1)
        logit2 = self.aux_head2(feat2)
        logit3 = self.aux_head3(feat3)
        logit4 = self.aux_head4(feat4)
        logit = self.head(self.bga(dfm, sfm))

        return [logit, logit1, logit2, logit3, logit4]

    def init_weight(self, pretrained=None):
        """
        Initialize the parameters of model parts.
        Args:
            pretrained ([str], optional): the path of pretrained model.. Defaults to None.
        """
        if pretrained is not None:
            if os.path.exists(pretrained):
                utils.load_pretrained_model(self, pretrained)
            else:
                raise Exception(
                    'Pretrained model is not found: {}'.format(pretrained))
        else:
            for sublayer in self.sublayers():
                if isinstance(sublayer, nn.Conv2d):
                    param_init.msra_init(sublayer.weight)
                elif isinstance(sublayer, nn.SyncBatchNorm):
                    param_init.constant_init(sublayer.weight, value=1.0)
                    param_init.constant_init(sublayer.bias, value=0.0)

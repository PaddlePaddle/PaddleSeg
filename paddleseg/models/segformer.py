# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers


class MLP(nn.Layer):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.proj(x)
        return x


@manager.MODELS.add_component
class SegFormer(nn.Layer):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 embedding_dim,
                 align_corners=False,
                 pretrained=None):
        super(SegFormer, self).__init__()

        self.backbone = backbone
        self.num_classes = num_classes
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.backbone.feat_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.dropout = nn.Dropout2D(0.1)
        self.linear_fuse = layers.ConvBNReLU(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            bias_attr=False)

        self.linear_pred = nn.Conv2D(
            embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        feats = self.backbone(x)
        c1, c2, c3, c4 = feats

        ############## MLP decoder on C1-C4 ###########
        c1_shape = paddle.shape(c1)
        c2_shape = paddle.shape(c2)
        c3_shape = paddle.shape(c3)
        c4_shape = paddle.shape(c4)

        _c4 = self.linear_c4(c4).transpose([0, 2, 1]).reshape(
            [0, 0, c4_shape[2], c4_shape[3]])
        _c4 = F.interpolate(
            _c4, size=c1_shape[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).transpose([0, 2, 1]).reshape(
            [0, 0, c3_shape[2], c3_shape[3]])
        _c3 = F.interpolate(
            _c3, size=c1_shape[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).transpose([0, 2, 1]).reshape(
            [0, 0, c2_shape[2], c2_shape[3]])
        _c2 = F.interpolate(
            _c2, size=c1_shape[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).transpose([0, 2, 1]).reshape(
            [0, 0, c1_shape[2], c1_shape[3]])

        _c = self.linear_fuse(paddle.concat([_c4, _c3, _c2, _c1], axis=1))

        logit = self.dropout(_c)
        logit = self.linear_pred(logit)
        return [
            F.interpolate(
                logit,
                size=paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=False)
        ]


@manager.MODELS.add_component
def SegFormer_B0(**kwargs):
    return SegFormer(
        backbone=manager.BACKBONES['MixVisionTransformer_B0'](),
        embedding_dim=256,
        **kwargs)


@manager.MODELS.add_component
def SegFormer_B1(**kwargs):
    return SegFormer(
        backbone=manager.BACKBONES['MixVisionTransformer_B1'](),
        embedding_dim=256,
        **kwargs)


@manager.MODELS.add_component
def SegFormer_B2(**kwargs):
    return SegFormer(
        backbone=manager.BACKBONES['MixVisionTransformer_B2'](),
        embedding_dim=768,
        **kwargs)


@manager.MODELS.add_component
def SegFormer_B3(**kwargs):
    return SegFormer(
        backbone=manager.BACKBONES['MixVisionTransformer_B3'](),
        embedding_dim=768,
        **kwargs)


@manager.MODELS.add_component
def SegFormer_B4(**kwargs):
    return SegFormer(
        backbone=manager.BACKBONES['MixVisionTransformer_B4'](),
        embedding_dim=768,
        **kwargs)


@manager.MODELS.add_component
def SegFormer_B5(**kwargs):
    return SegFormer(
        backbone=manager.BACKBONES['MixVisionTransformer_B5'](),
        embedding_dim=768,
        **kwargs)

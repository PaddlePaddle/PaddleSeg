# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.nn as nn
import paddle.nn.functional as F

import paddle
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.utils import utils


@manager.MODELS.add_component
class SegmentationTransformer(nn.Layer):
    '''
    The SETR implementation based on PaddlePaddle.

    The original article refers to
        Zheng, Sixiao, et al. "Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers"
        (https://arxiv.org/abs/2012.15840)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): Backbone network.
        backbone_indices (tuple): A tuple indicates the indices of output of backbone.
            It can be either one or two values, if two values, the first index will be taken as
            a deep-supervision feature in auxiliary layer; the second one will be taken as
            input of pixel representation. If one value, it is taken by both above.
        head (str, optional): SETR head type(naive, pup or mla). Default: naive.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    '''

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(9, 14, 19, 23),
                 head='naive',
                 align_corners=False,
                 pretrained=None,
                 **head_config):

        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        if head.lower() == 'naive':
            self.head = NaiveHead(
                num_classes=self.num_classes,
                backbone_indices=backbone_indices,
                in_channels=self.backbone.embed_dim,
                **head_config)
        elif head.lower() == 'pup':
            self.head = PUPHead(
                num_classes=self.num_classes,
                backbone_indices=backbone_indices,
                align_corners=align_corners,
                in_channels=self.backbone.embed_dim,
                **head_config)
        elif head.lower() == 'mla':
            self.head = MLAHead(
                num_classes=self.num_classes,
                backbone_indices=backbone_indices,
                in_channels=self.backbone.embed_dim,
                **head_config)
        else:
            raise RuntimeError(
                'Unsupported segmentation head type {}. Only naive/pup/mla is valid.'
                .format(head))

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        x_shape = paddle.shape(x)
        feats, _shape = self.backbone(x)
        logits = self.head(feats, _shape)
        return [
            F.interpolate(
                _logit,
                x_shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for _logit in logits
        ]


class NaiveHead(nn.Layer):
    '''
    The SETR Naive Head implementation.

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): A tuple indicates the indices of output of backbone.
            It can be either one or two values, if two values, the first index will be taken as
            a deep-supervision feature in auxiliary layer; the second one will be taken as
            input of pixel representation. If one value, it is taken by both above.
        in_channels (int): The number of input channels. Default: 10.
        lr_multiple (int, optional): The leanring rate multiple of head parameters. Default: 10.
    '''

    def __init__(self,
                 num_classes,
                 backbone_indices,
                 in_channels,
                 lr_multiple=10):
        super().__init__()

        self.cls_head_norm = nn.LayerNorm(
            normalized_shape=in_channels, epsilon=1e-6)
        self.cls_head = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=in_channels, out_channels=256, kernel_size=1),
            nn.Conv2D(
                in_channels=256, out_channels=num_classes, kernel_size=1))

        aux_head_nums = len(backbone_indices) - 1
        self.aux_head_norms = nn.LayerList(
            [nn.LayerNorm(
                normalized_shape=in_channels, epsilon=1e-6)] * aux_head_nums)
        self.aux_heads = nn.LayerList([
            nn.Sequential(
                layers.ConvBNReLU(
                    in_channels=in_channels, out_channels=256, kernel_size=1),
                nn.Conv2D(
                    in_channels=256, out_channels=num_classes, kernel_size=1))
        ] * aux_head_nums)

        self.in_channels = in_channels
        self.lr_multiple = lr_multiple
        self.backbone_indices = backbone_indices
        self.init_weight()

    def init_weight(self):
        for _param in self.parameters():
            _param.optimize_attr['learning_rate'] = self.lr_multiple

        for layer in self.sublayers():
            if isinstance(layer, nn.LayerNorm):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)

    def forward(self, x, _shape):
        logits = []
        feat = x[self.backbone_indices[-1]]
        feat = self.cls_head_norm(feat).transpose([0, 2, 1]).reshape(
            [0, self.in_channels, _shape[2], _shape[3]])

        logits.append(self.cls_head(feat))

        if self.training:
            for idx, _head in enumerate(self.aux_heads):
                feat = x[self.backbone_indices[idx]]
                feat = self.aux_head_norms[idx](feat).transpose(
                    [0, 2,
                     1]).reshape([0, self.in_channels, _shape[2], _shape[3]])
                logits.append(_head(feat))

        return logits


class PUPHead(nn.Layer):
    '''
    The SETR Progressive UPsampling Head implementation.

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): A tuple indicates the indices of output of backbone.
            It can be either one or two values, if two values, the first index will be taken as
            a deep-supervision feature in auxiliary layer; the second one will be taken as
            input of pixel representation. If one value, it is taken by both above.
        in_channels (int): The number of input channels. Default: 10.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        lr_multiple (int, optional): The leanring rate multiple of head parameters. Default: 10.
    '''

    def __init__(self,
                 num_classes,
                 backbone_indices,
                 in_channels,
                 align_corners=False,
                 lr_multiple=10):
        super().__init__()

        inter_channels = 256

        self.cls_head_norm = nn.LayerNorm(
            normalized_shape=in_channels, epsilon=1e-6)
        self.cls_head = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=3,
                padding=1),
            nn.Upsample(
                scale_factor=2, mode='bilinear'),
            layers.ConvBNReLU(
                in_channels=inter_channels,
                out_channels=inter_channels,
                kernel_size=3,
                padding=1),
            nn.Upsample(
                scale_factor=2, mode='bilinear'),
            layers.ConvBNReLU(
                in_channels=inter_channels,
                out_channels=inter_channels,
                kernel_size=3,
                padding=1),
            nn.Upsample(
                scale_factor=2, mode='bilinear'),
            layers.ConvBNReLU(
                in_channels=inter_channels,
                out_channels=inter_channels,
                kernel_size=3,
                padding=1),
            nn.Conv2D(
                in_channels=inter_channels,
                out_channels=num_classes,
                kernel_size=1))

        aux_head_nums = len(backbone_indices)
        self.aux_head_norms = nn.LayerList(
            [nn.LayerNorm(
                normalized_shape=in_channels, epsilon=1e-6)] * aux_head_nums)
        self.aux_heads = nn.LayerList([
            nn.Sequential(
                layers.ConvBNReLU(
                    in_channels=in_channels,
                    out_channels=inter_channels,
                    kernel_size=3,
                    padding=1),
                nn.Upsample(
                    scale_factor=4, mode='bilinear'),
                nn.Conv2D(
                    in_channels=inter_channels,
                    out_channels=num_classes,
                    kernel_size=1))
        ] * aux_head_nums)

        self.in_channels = in_channels
        self.lr_multiple = lr_multiple
        self.backbone_indices = backbone_indices
        self.init_weight()

    def init_weight(self):
        for _param in self.parameters():
            _param.optimize_attr['learning_rate'] = self.lr_multiple

        for layer in self.sublayers():
            if isinstance(layer, nn.LayerNorm):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)

    def forward(self, x, _shape):
        logits = []
        feat = x[self.backbone_indices[-1]]
        feat = self.cls_head_norm(feat).transpose([0, 2, 1]).reshape(
            [0, self.in_channels, _shape[2], _shape[3]])

        logits.append(self.cls_head(feat))

        if self.training:
            for idx, _head in enumerate(self.aux_heads):
                feat = x[self.backbone_indices[idx]]
                feat = self.aux_head_norms[idx](feat).transpose(
                    [0, 2,
                     1]).reshape([0, self.in_channels, _shape[2], _shape[3]])
                logits.append(_head(feat))

        return logits


class ConvMLA(nn.Layer):
    def __init__(self, in_channels, mla_channels):
        super().__init__()

        self.mla_p2_1x1 = layers.ConvBNReLU(
            in_channels=in_channels, out_channels=mla_channels, kernel_size=1)

        self.mla_p3_1x1 = layers.ConvBNReLU(
            in_channels=in_channels, out_channels=mla_channels, kernel_size=1)

        self.mla_p4_1x1 = layers.ConvBNReLU(
            in_channels=in_channels, out_channels=mla_channels, kernel_size=1)

        self.mla_p5_1x1 = layers.ConvBNReLU(
            in_channels=in_channels, out_channels=mla_channels, kernel_size=1)

        self.mla_p2 = layers.ConvBNReLU(
            in_channels=mla_channels,
            out_channels=mla_channels,
            kernel_size=3,
            padding=1)

        self.mla_p3 = layers.ConvBNReLU(
            in_channels=mla_channels,
            out_channels=mla_channels,
            kernel_size=3,
            padding=1)

        self.mla_p4 = layers.ConvBNReLU(
            in_channels=mla_channels,
            out_channels=mla_channels,
            kernel_size=3,
            padding=1)

        self.mla_p5 = layers.ConvBNReLU(
            in_channels=mla_channels,
            out_channels=mla_channels,
            kernel_size=3,
            padding=1)

    def forward(self, x):
        res2, res3, res4, res5 = x

        mla_p5_1x1 = self.mla_p5_1x1(res5)
        mla_p4_1x1 = self.mla_p4_1x1(res4)
        mla_p3_1x1 = self.mla_p3_1x1(res3)
        mla_p2_1x1 = self.mla_p2_1x1(res2)

        mla_p4_plus = mla_p5_1x1 + mla_p4_1x1
        mla_p3_plus = mla_p4_plus + mla_p3_1x1
        mla_p2_plus = mla_p3_plus + mla_p2_1x1

        mla_p5 = self.mla_p5(mla_p5_1x1)
        mla_p4 = self.mla_p4(mla_p4_plus)
        mla_p3 = self.mla_p3(mla_p3_plus)
        mla_p2 = self.mla_p2(mla_p2_plus)

        return [mla_p2, mla_p3, mla_p4, mla_p5]


class MLAHead(nn.Layer):
    '''
    The SETR Multi-Level feature Aggregation Head implementation.

    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): A tuple indicates the indices of output of backbone.
            It can be either one or two values, if two values, the first index will be taken as
            a deep-supervision feature in auxiliary layer; the second one will be taken as
            input of pixel representation. If one value, it is taken by both above.
        in_channels (int): The number of input channels. Default: 10.
        mla_channels (int, optional): The number of middle channels of ConvMLA Layer. Default: 256.
        mlahead_channels (int, optional): The number of middle channels of mla head. Default: 128.
        lr_multiple (int, optional): The leanring rate multiple of head parameters. Default: 10.
    '''

    def __init__(self,
                 num_classes,
                 backbone_indices,
                 in_channels,
                 mla_channels=256,
                 mlahead_channels=128,
                 lr_multiple=10):
        super().__init__()

        if len(backbone_indices) != 4:
            raise RuntimeError

        self.mla_feat_nums = len(backbone_indices)
        self.norms = nn.LayerList(
            [nn.LayerNorm(
                normalized_shape=in_channels,
                epsilon=1e-6)] * self.mla_feat_nums)

        self.mla = ConvMLA(in_channels, mla_channels)

        self.aux_heads = nn.LayerList([
            nn.Conv2D(
                in_channels=mla_channels,
                out_channels=num_classes,
                kernel_size=1)
        ] * self.mla_feat_nums)

        self.feat_convs = nn.LayerList([
            nn.Sequential(
                layers.ConvBNReLU(
                    in_channels=mla_channels,
                    out_channels=mlahead_channels,
                    kernel_size=3,
                    padding=1),
                layers.ConvBNReLU(
                    in_channels=mlahead_channels,
                    out_channels=mlahead_channels,
                    kernel_size=3,
                    padding=1),
                nn.Upsample(
                    scale_factor=4, mode='bilinear', align_corners=True))
        ] * self.mla_feat_nums)

        self.backbone_indices = backbone_indices
        self.in_channels = in_channels

        self.cls_head = nn.Conv2D(
            in_channels=4 * mlahead_channels,
            out_channels=num_classes,
            kernel_size=3,
            padding=1)

    def init_weight(self):
        for name, _param in self.named_parameters():
            if name.startswith('norms.') or name.startswith('mla.'):
                continue

            _param.optimize_attr['learning_rate'] = self.lr_multiple

        for layer in self.sublayers():
            if isinstance(layer, nn.LayerNorm):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)

    def forward(self, x, _shape):
        logits = []

        feats = [x[_idx] for _idx in self.backbone_indices]

        for i in range(self.mla_feat_nums):
            feats[i] = self.norms[i](feats[i]).transpose([0, 2, 1]).reshape(
                [0, self.in_channels, _shape[2], _shape[3]])

        feats = self.mla(feats)
        if self.training:
            for i in range(self.mla_feat_nums):
                logits.append(self.aux_heads[i](feats[i]))

        for i in range(self.mla_feat_nums):
            feats[i] = self.feat_convs[i](feats[i])

        feat_mix = paddle.concat(feats, axis=1)
        logits.insert(0, self.cls_head(feat_mix))

        return logits

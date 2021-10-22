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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager, param_init
from paddleseg.utils import utils
from paddleseg.models.backbones.vision_transformer import Block
from paddleseg.models.backbones.transformer_utils import *

__all__ = ['Segmenter', 'SegmenterLinearDecoder', 'SegmenterMaskTransformerDecoder']

@manager.MODELS.add_component
class Segmenter(nn.Layer):
    '''
    The Segmenter implementation based on PaddlePaddle.

    The original article refers to
    Strudel, Robin, et al. "Segmenter: Transformer for Semantic Segmentation."
    arXiv preprint arXiv:2105.05633 (2021).

    Args:
        backbone (Paddle.nn.Layer): Backbone transformer network.
        head (Paddle.nn.Layer): Head network, such as mask_transformer.
        num_classes (int): The unique number of target classes. Default: None.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    '''

    def __init__(self, backbone, head, num_classes=None, pretrained=None):

        super().__init__()
        self.backbone = backbone
        self.head = head

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        x_shape = paddle.shape(x)

        feats, _shape = self.backbone(x)
        logits = self.head(feats[-1], _shape[2:])

        return [
            F.interpolate(_logit, x_shape[2:], mode='bilinear')
            for _logit in logits
        ]

@manager.MODELS.add_component
class SegmenterLinearDecoder(nn.Layer):
    """ The linear decoder of Segmenter.
    Args:
        in_dim (int): The embed dim of input.
        num_classes (int): The unique number of target classes.
    """
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(in_dim, num_classes)
        self.apply(init_weights)

    def forward(self, x, patch_embed_size):
        """ Forward function.
        Args:
            x (Tensor): Input tensor of decoder.
            patch_embed_size (list): The size of patch embed tensor, such as (n, h, w, c).
        Returns:
            list[Tensor]: Segmentation results.
        """
        masks = self.head(x)
        
        #[b, (h w), c] -> [b, c, h, w]
        h, w = patch_embed_size
        masks = masks.reshape((0, h, w, paddle.shape(masks)[-1]))
        masks = masks.transpose((0, 3, 1, 2))

        return [masks]


@manager.MODELS.add_component
class SegmenterMaskTransformerDecoder(nn.Layer):
    """ The Mask Transformer Decoder of Segmenter.
    Args:
        num_classes (int): The unique number of target classes.
        in_dim (int): The embed dim of input.
        embed_dim (int): Embedding dim of mask transformer.
        depth (int): The num of layers in Transformer.
        num_heads (int): The num of heads in MSA.
        mlp_ratio (int): Ratio of MLP dim.
        drop_rate (float): Drop rate of MLP in MSA.
        drop_path_rate (float): Drop path rate in MSA.
        attn_drop_rate (float): Attenation drop rate in MSA.
        qkv_bias (bool): Whether add bias in qkv linear.
    """

    def __init__(self,
                 num_classes,
                 in_dim,
                 embed_dim,
                 depth,
                 num_heads,
                 mlp_ratio=4,
                 drop_rate=0.0,
                 drop_path_rate=0.0,
                 attn_drop_rate=0.0,
                 qkv_bias=False):
        super().__init__()
        self.num_classes = num_classes

        self.proj_input = nn.Linear(in_dim, embed_dim)

        self.cls_token = self.create_parameter(
            shape=(1, num_classes, embed_dim),
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))

        dpr = [x for x in np.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[i],
                attn_drop=attn_drop_rate,
                qkv_bias=qkv_bias) for i in range(depth)
        ])

        initializer = paddle.nn.initializer.TruncatedNormal(std=0.02)
        self.proj_patch = nn.Linear(
            embed_dim,
            embed_dim,
            weight_attr=paddle.ParamAttr(initializer=initializer),
            bias_attr=False)
        self.proj_class = nn.Linear(
            embed_dim,
            embed_dim,
            weight_attr=paddle.ParamAttr(initializer=initializer),
            bias_attr=False)

        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.mask_norm = nn.LayerNorm(num_classes)

        self.apply(init_weights)

    def forward(self, x, patch_embed_size):
        """ Forward function.
        Args:
            x (Tensor): Input tensor of decoder.
            patch_embed_size (list): The size of patch embed tensor, such as (n, h, w, c).
        Returns:
            list[Tensor]: Segmentation results.
        """
        x = self.proj_input(x)

        cls_token = self.cls_token.expand((x.shape[0], -1, -1))
        x = paddle.concat([x, cls_token], axis=1)

        for block in self.blocks:
            x = block(x)
        x = self.decoder_norm(x)

        patches, masks = x[:, :-self.num_classes], x[:, -self.num_classes:]
        patches = self.proj_patch(patches)
        masks = self.proj_class(masks)
        patches = patches / paddle.norm(patches, axis=-1, keepdim=True)
        masks = masks / paddle.norm(masks, axis=-1, keepdim=True)

        masks = patches @ masks.transpose((0, 2, 1))
        masks = self.mask_norm(masks)

        #[b, (h w), c] -> [b, c, h, w]
        h, w = patch_embed_size
        masks = masks.reshape((0, h, w, paddle.shape(masks)[-1]))
        masks = masks.transpose((0, 3, 1, 2))

        return [masks]


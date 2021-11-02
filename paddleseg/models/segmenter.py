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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from paddleseg.utils import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models.backbones import vision_transformer, transformer_utils

__all__ = ['LinearSegmenter', 'MaskSegmenter']


@manager.MODELS.add_component
class LinearSegmenter(nn.Layer):
    '''
    The implementation of segmenter with linear head based on PaddlePaddle.

    The original article refers to Strudel, Robin, et al. "Segmenter: Transformer
    for Semantic Segmentation." arXiv preprint arXiv:2105.05633 (2021).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (nn.Layer): The backbone transformer network.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    '''

    def __init__(self, num_classes, backbone, pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.head = SegmenterLinearHead(num_classes, backbone.embed_dim)
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        x_shape = paddle.shape(x)

        feats, shape = self.backbone(x)
        logits = self.head(feats[-1], shape[2:])

        logit_list = [
            F.interpolate(logit, x_shape[2:], mode='bilinear')
            for logit in logits
        ]

        return logit_list


@manager.MODELS.add_component
class MaskSegmenter(nn.Layer):
    '''
    The implementation of segmenter with mask head based on PaddlePaddle.

    The original article refers to Strudel, Robin, et al. "Segmenter: Transformer
    for Semantic Segmentation." arXiv preprint arXiv:2105.05633 (2021).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (nn.Layer): The backbone transformer network.
        h_embed_dim (int): The embedding dim in mask head.
        h_depth (int): The num of layers in mask head.
        h_num_heads (int): The num of heads of MSA in mask head.
        h_mlp_ratio (int, optional): Ratio of MLP dim in mask head. Default: 4.
        h_drop_rate (float, optional): Drop rate of MLP in mask head. Default: 0.0.
        h_drop_path_rate (float, optional): Drop path rate in mask head. Default: 0.0.
        h_attn_drop_rate (float, optional): Attenation drop rate in mask head. Default: 0.0.
        h_qkv_bias (bool, optional): Whether add bias in mask head. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    '''

    def __init__(self,
                 num_classes,
                 backbone,
                 h_embed_dim,
                 h_depth,
                 h_num_heads,
                 h_mlp_ratio=4,
                 h_drop_rate=0.0,
                 h_drop_path_rate=0.0,
                 h_attn_drop_rate=0.0,
                 h_qkv_bias=False,
                 pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.head = SegmenterMaskHead(
            num_classes, backbone.embed_dim, h_embed_dim, h_depth, h_num_heads,
            h_mlp_ratio, h_drop_rate, h_drop_path_rate, h_attn_drop_rate,
            h_qkv_bias)
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        x_shape = paddle.shape(x)

        feats, shape = self.backbone(x)
        logits = self.head(feats[-1], shape[2:])

        logit_list = [
            F.interpolate(logit, x_shape[2:], mode='bilinear')
            for logit in logits
        ]

        return logit_list


class SegmenterLinearHead(nn.Layer):
    '''
    The linear head of Segmenter.
    Args:
        num_classes (int): The unique number of target classes.
        in_dim (int): The embed dim of input.
    '''

    def __init__(self, num_classes, in_dim):
        super().__init__()
        self.head = nn.Linear(in_dim, num_classes)
        self.apply(transformer_utils.init_weights)

    def forward(self, x, patch_embed_size):
        """ Forward function.
        Args:
            x (Tensor): Input tensor of decoder.
            patch_embed_size (Tensor): The height and width of the patch embed tensor.
        Returns:
            list[Tensor]: Segmentation results.
        """
        masks = self.head(x)

        #[b, (h w), c] -> [b, c, h, w]
        h, w = patch_embed_size[0], patch_embed_size[1]
        masks = masks.reshape((0, h, w, paddle.shape(masks)[-1]))
        masks = masks.transpose((0, 3, 1, 2))

        return [masks]


class SegmenterMaskHead(nn.Layer):
    '''
    The mask head of segmenter.
    Args:
        num_classes (int): The unique number of target classes.
        in_dim (int): The embed dim of input.
        embed_dim (int): Embedding dim of mask transformer.
        depth (int): The num of layers in Transformer.
        num_heads (int): The num of heads in MSA.
        mlp_ratio (int, optional): Ratio of MLP dim. Default: 4.
        drop_rate (float, optional): Drop rate of MLP in MSA. Default: 0.0.
        drop_path_rate (float, optional): Drop path rate in MSA. Default: 0.0.
        attn_drop_rate (float, optional): Attenation drop rate in MSA. Default: 0.0.
        qkv_bias (bool, optional): Whether add bias in qkv linear. Default: False.
    '''

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
            vision_transformer.Block(
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

        self.apply(transformer_utils.init_weights)

    def forward(self, x, patch_embed_size):
        """ Forward function.
        Args:
            x (Tensor): Input tensor of decoder.
            patch_embed_size (Tensor): The height and width of the patch embed tensor.
        Returns:
            list[Tensor]: Segmentation results.
        """
        x = self.proj_input(x)

        cls_token = self.cls_token.expand((paddle.shape(x)[0], -1, -1))
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
        masks = masks.reshape((0, 0,
                               self.num_classes))  # For export inference model
        masks = self.mask_norm(masks)

        #[b, (h w), c] -> [b, c, h, w]
        h, w = patch_embed_size[0], patch_embed_size[1]
        masks = masks.reshape((0, h, w, paddle.shape(masks)[-1]))
        masks = masks.transpose((0, 3, 1, 2))

        return [masks]

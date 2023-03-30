# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddleseg.utils import utils

from paddlepanseg.cvlibs import manager
from paddlepanseg.cvlibs import build_info_dict
from .pixel_decoder import MSDeformAttnPixelDecoder
from .transformer_decoder import MultiScaleMaskedTransformerDecoder

__all__ = ['Mask2Former']


@manager.MODELS.add_component
class Mask2Former(nn.Layer):
    """
    The Mask2Former implementation based on PaddlePaddle.

    The original article refers to
     Bowen Cheng, et, al. "Masked-attention Mask Transformer for Universal Image Segmentation"
     (https://arxiv.org/abs/2112.01527)

    Args:
        num_classes (int): The number of target semantic classes.
        backbone (paddle.nn.Layer): The backbone network. Currently supports ResNet50-vd/ResNet101-vd/Xception65.
        backbone_indices (tuple|None): The indices of backbone output feature maps to use.
        backbone_feat_os (tuple|None): The output strides of backbone output feature maps.
        num_queries (int): The number of queries to use in the decoder.
        pd_num_heads (int): The number of heads of the multi-head attention modules used in the pixel decoder.
        pd_conv_dim (int): The number of convolutional filters used for input projection in the pixel decoder.
        pd_mask_dim (int): The number of convolutional filters used to produce mask features in the pixel decoder.
        pd_ff_dim (int): The number of feature channels in the feed-forward networks used in the pixel decoder.
        pd_num_layers (int): The number of basic layers used in the pixel decoder.
        pd_common_stride (int): The base output stride of feature maps in the pixel decoder.
        td_hidden_dim (int): The dimension of the hidden features in the transformer decoder.
        td_num_head (int): The number of heads of the multi-head attention modules used in the transformer decoder.
        td_ff_dim (int): The number of feature channels in the feed-forward networks used in the transformer decoder.
        td_num_layers (int): The number of basic layers used in the transformer decoder.
        td_pre_norm (bool): Whether or not to normalize features before the attention operation in the transformer 
            decoder.
        td_mask_dim (bool): The number of convolutional filters used for mask prediction in the transformer decoder.
        td_enforce_proj (bool): Whether or not to use an additional input projection layer in the transformer decoder.
        pretrained (str|None, optional): The path or url of pretrained model. If None, no pretrained model will be used.
            Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices,
                 backbone_feat_os,
                 num_queries,
                 pd_num_heads,
                 pd_conv_dim,
                 pd_mask_dim,
                 pd_ff_dim,
                 pd_num_layers,
                 pd_common_stride,
                 td_hidden_dim,
                 td_num_heads,
                 td_ff_dim,
                 td_num_layers,
                 td_pre_norm,
                 td_mask_dim,
                 td_enforce_proj,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        self.num_queries = num_queries
        self.pixel_decoder = MSDeformAttnPixelDecoder(
            in_feat_strides=backbone_feat_os,
            in_feat_chns=self.backbone.feat_channels,
            feat_indices=backbone_indices,
            num_heads=pd_num_heads,
            ff_dim=pd_ff_dim,
            num_enc_layers=pd_num_layers,
            conv_dim=pd_conv_dim,
            mask_dim=pd_mask_dim,
            common_stride=pd_common_stride)
        self.transformer_decoder = MultiScaleMaskedTransformerDecoder(
            in_channels=pd_conv_dim,
            num_classes=num_classes,
            hidden_dim=td_hidden_dim,
            num_queries=self.num_queries,
            num_heads=td_num_heads,
            ff_dim=td_ff_dim,
            num_dec_layers=td_num_layers,
            pre_norm=td_pre_norm,
            mask_dim=td_mask_dim,
            enforce_input_proj=td_enforce_proj)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        features = self.backbone(x)
        multi_scale_features, mask_features = self.pixel_decoder(features)
        pred_logits, pred_masks, aux_logits, aux_masks = self.transformer_decoder(
            multi_scale_features, mask_features)
        res = build_info_dict(
            _type_='net_out', logits=pred_logits, masks=pred_masks)
        res['map_fields'] = ['masks']
        if self.training:
            res['aux_logits'] = aux_logits
            res['aux_masks'] = aux_masks
        else:
            res['masks'] = F.interpolate(
                res['masks'],
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False)
        return res

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

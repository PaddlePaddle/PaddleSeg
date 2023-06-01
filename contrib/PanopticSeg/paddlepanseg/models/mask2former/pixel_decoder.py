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

import copy

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import param_init

from paddlepanseg.models.param_init import c2_xavier_fill, THLinearInitMixin
from .common import Conv2D, PositionEmbeddingSine
from ._ms_deform_attn import MSDeformAttn


class MSDeformAttnTransformerEncoderOnly(nn.Layer, THLinearInitMixin):
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 num_encoder_layers=6,
                 ff_dim=1024,
                 dropout=0.1,
                 num_feature_levels=4,
                 enc_num_points=4):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            embed_dim, ff_dim, dropout, num_feature_levels, num_heads,
            enc_num_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer,
                                                      num_encoder_layers)

        self.level_embed = self.create_parameter(
            shape=[num_feature_levels, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Uniform())

        self.init_weight()

    def init_weight(self):
        super().init_weight()
        for p in self.parameters():
            if p.dim() > 1:
                param_init.xavier_uniform(p)
        for m in self.sublayers():
            if isinstance(m, MSDeformAttn):
                m.init_weight()
        param_init.normal_init(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = paddle.sum(~mask[:, :, 0], 1)
        valid_W = paddle.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.astype('float32') / H
        valid_ratio_w = valid_W.astype('float32') / W
        valid_ratio = paddle.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [
            paddle.zeros(
                (x.shape[0], x.shape[2], x.shape[3]), dtype='bool')
            for x in srcs
        ]
        # Prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask,
                  pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose((0, 2, 1))
            mask = mask.reshape((mask.shape[0], -1))
            pos_embed = pos_embed.flatten(2).transpose((0, 2, 1))
            lvl_pos_embed = pos_embed + self.level_embed[lvl].reshape(
                (1, 1, -1))
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = paddle.concat(src_flatten, 1)
        mask_flatten = paddle.concat(mask_flatten, 1)
        lvl_pos_embed_flatten = paddle.concat(lvl_pos_embed_flatten, 1)
        spatial_shapes = paddle.to_tensor(spatial_shapes, dtype='int64')
        level_start_index = paddle.concat(
            (paddle.zeros(
                (1, ), dtype='int64'),
             spatial_shapes.prod(1).cumsum(0)[:-1])).astype('int64')
        valid_ratios = paddle.stack([self.get_valid_ratio(m) for m in masks], 1)

        # Encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index,
                              valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 embed_dim=256,
                 ffn_dim=1024,
                 dropout=0.1,
                 num_levels=4,
                 num_heads=8,
                 num_points=4):
        super().__init__()

        # Self attention
        self.self_attn = MSDeformAttn(embed_dim, num_levels, num_heads,
                                      num_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # FFN
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self,
                src,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask=None):
        # Self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos), reference_points, src,
            spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Layer):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.LayerList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = paddle.meshgrid(
                paddle.linspace(0.5, H_ - 0.5, H_.astype('int32'), 'float32'),
                paddle.linspace(0.5, W_ - 0.5, W_.astype('int32'), 'float32'))
            ref_y = ref_y.flatten()[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.flatten()[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = paddle.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = paddle.concat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self,
                src,
                spatial_shapes,
                level_start_index,
                valid_ratios,
                pos=None,
                padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes,
                                                     valid_ratios)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes,
                           level_start_index, padding_mask)

        return output


class MSDeformAttnPixelDecoder(nn.Layer):
    def __init__(self, num_heads, ff_dim, num_enc_layers, conv_dim, mask_dim,
                 in_feat_strides, in_feat_chns, feat_indices, common_stride):
        super().__init__()

        self.feature_strides = in_feat_strides
        self.feature_channels = in_feat_chns
        self.transformer_indices = feat_indices
        self.transformer_strides = [
            self.feature_strides[i] for i in self.transformer_indices
        ]
        self.transformer_channels = [
            self.feature_channels[i] for i in self.transformer_indices
        ]

        self.num_feature_levels = len(self.transformer_indices)
        input_proj_list = []
        # From low resolution to high resolution (res5 -> res2)
        for in_channels in self.transformer_channels[::-1]:
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(
                        num_channels=conv_dim, num_groups=32), ))
        self.input_proj = nn.LayerList(input_proj_list)

        for proj in self.input_proj:
            # Note that we do not set gain=1 here, which differs from the original impl
            param_init.xavier_uniform(proj[0].weight)
            param_init.constant_init(proj[0].bias, value=0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            embed_dim=conv_dim,
            dropout=0.0,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_encoder_layers=num_enc_layers,
            num_feature_levels=self.num_feature_levels, )
        num_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(num_steps, normalize=True)

        self.mask_dim = mask_dim
        # Use 1x1 conv instead
        self.mask_conv = Conv2D(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0, )
        c2_xavier_fill(self.mask_conv)

        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # Extra fpn levels
        stride = min(self.transformer_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        for idx, in_channels in enumerate(
                self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = nn.GroupNorm(num_channels=conv_dim, num_groups=32)
            output_norm = nn.GroupNorm(num_channels=conv_dim, num_groups=32)

            lateral_conv = Conv2D(
                in_channels,
                conv_dim,
                kernel_size=1,
                bias_attr=False,
                norm=lateral_norm)
            output_conv = Conv2D(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False,
                norm=output_norm,
                activation=nn.ReLU(), )
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)

            self.add_sublayer("adapter_{}".format(idx + 1), lateral_conv)
            self.add_sublayer("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def forward(self, features):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, ti in enumerate(self.transformer_indices[::-1]):
            x = features[ti].astype(
                'float32')  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.num_feature_levels
        for i in range(self.num_feature_levels):
            if i < self.num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[
                    i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = paddle.split(y, split_size_or_sections, axis=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(
                z.transpose((0, 2, 1)).reshape((bs, -1, spatial_shapes[i][0],
                                                spatial_shapes[i][1])))

        # Append `out` with extra FPN levels
        for idx, x in enumerate(features[:self.num_fpn_levels][::-1]):
            x = x.astype('float32')
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(
                out[-1],
                size=cur_fpn.shape[-2:],
                mode='bilinear',
                align_corners=False)
            y = output_conv(y)
            out.append(y)

        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return multi_scale_features, self.mask_conv(out[-1])

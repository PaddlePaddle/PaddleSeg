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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import param_init

from paddlepanseg.models.param_init import c2_xavier_fill, THLinearInitMixin, th_linear_fill, th_multihead_fill
from .common import Conv2D, PositionEmbeddingSine


class SelfAttentionLayer(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(
            embed_dim, num_heads, dropout=dropout)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self.init_weight()

    def init_weight(self):
        th_multihead_fill(self.self_attn, True)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_mask, query_pos):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, tgt_mask, query_pos):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, tgt_mask, query_pos):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, query_pos)


class CrossAttentionLayer(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiHeadAttention(
            embed_dim, num_heads, dropout=dropout)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self.init_weight()

    def init_weight(self):
        th_multihead_fill(self.multihead_attn, True)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, memory_mask, pos, query_pos):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory, memory_mask, pos, query_pos):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask)
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory, memory_mask, pos, query_pos):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, pos, query_pos)


class FFNLayer(nn.Layer, THLinearInitMixin):
    def __init__(self,
                 embed_dim,
                 ff_dim=2048,
                 dropout=0.0,
                 normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self.init_weight()

    def init_weight(self):
        super().init_weight()
        for p in self.parameters():
            if p.dim() > 1:
                param_init.xavier_uniform(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(nn.Layer, THLinearInitMixin):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.init_weight()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiScaleMaskedTransformerDecoder(nn.Layer):
    def __init__(
            self,
            in_channels,
            num_classes,
            hidden_dim,
            num_queries,
            num_heads,
            ff_dim,
            num_dec_layers,
            pre_norm,
            mask_dim,
            enforce_input_proj, ):
        super().__init__()

        # Positional encoding
        n_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(n_steps, normalize=True)

        # Define Transformer decoder here
        self.num_heads = num_heads
        self.num_layers = num_dec_layers
        self.transformer_self_attention_layers = nn.LayerList()
        self.transformer_cross_attention_layers = nn.LayerList()
        self.transformer_ffn_layers = nn.LayerList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    normalize_before=pre_norm, ))

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    normalize_before=pre_norm, ))

            self.transformer_ffn_layers.append(
                FFNLayer(
                    embed_dim=hidden_dim,
                    ff_dim=ff_dim,
                    dropout=0.0,
                    normalize_before=pre_norm, ))

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # Learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # Learnable query
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.LayerList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_proj:
                self.input_proj.append(
                    Conv2D(
                        in_channels, hidden_dim, kernel_size=1))
                c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # Output FFNs
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.init_weight()

    def init_weight(self):
        param_init.normal_init(self.query_feat.weight)
        param_init.normal_init(self.query_embed.weight)
        param_init.normal_init(self.level_embed.weight)
        th_linear_fill(self.class_embed)

    def forward(self, x, mask_features):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) +
                       self.level_embed.weight[i][None, :, None])
            pos[-1] = pos[-1].transpose((0, 2, 1))
            src[-1] = src[-1].transpose((0, 2, 1))

        bs = src[0].shape[0]

        # QxNxC
        query_embed = paddle.tile(
            self.query_embed.weight.unsqueeze(0), (bs, 1, 1))
        output = paddle.tile(self.query_feat.weight.unsqueeze(0), (bs, 1, 1))

        predictions_class = []
        predictions_mask = []

        # Prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # XXX: Since Paddle boolean tensor does not support masked assign
            attn_mask = attn_mask.astype('int32')
            all_zero_mask = attn_mask.sum(-1, keepdim=True) == 0
            all_zero_mask = paddle.tile(all_zero_mask.detach(),
                                        (1, 1, attn_mask.shape[-1]))
            # If a query associates with a all-zero mask, do not use masked attention
            attn_mask = paddle.where(all_zero_mask,
                                     paddle.ones_like(attn_mask), attn_mask)
            attn_mask = attn_mask.astype('bool')

            # NOTE: cross-attention first
            # In paddle.nn.MultiHeadAttention, attn_mask==0 indicates an unwanted position
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                pos=pos[level_index],
                query_pos=query_embed)

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, query_pos=query_embed)

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) %
                                                self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        return predictions_class[-1], predictions_mask[
            -1], predictions_class[:-1], predictions_mask[:-1]

    def forward_prediction_heads(self, output, mask_features,
                                 attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output).astype('float32')
        mask_features = mask_features.astype('float32')
        outputs_mask = paddle.einsum('bqc,bchw->bqhw', mask_embed,
                                     mask_features)

        with paddle.no_grad():
            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(
                outputs_mask,
                size=attn_mask_target_size,
                mode='bilinear',
                align_corners=False)
            attn_mask = (paddle.tile(
                F.sigmoid(attn_mask).flatten(2).unsqueeze(1),
                (1, self.num_heads, 1, 1)) >= 0.5).astype('bool')

        return outputs_class, outputs_mask, attn_mask

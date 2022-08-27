# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/Beckschen/TransUNet/blob/main/networks/vit_seg_modeling.py

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import logging
import math
from os.path import join as pjoin

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from medicalseg.cvlibs import manager
from medicalseg.utils import load_pretrained_model

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return paddle.to_tensor(weights)


def swish(x):
    return x * F.sigmoid(x)


ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": swish}


class Attention(nn.Layer):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

        self.softmax = nn.Softmax(axis=-1)

    def transpose_for_scores(self, x):
        new_x_shape = list(x.shape[:-1]) + [
            self.num_attention_heads, self.attention_head_size
        ]
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3])

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = paddle.matmul(query_layer,
                                         key_layer.transpose([0, 1, 3, 2]))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = list(
            context_layer.shape[:-2]) + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Layer):
    def __init__(self, hidden_size, mlp_dim, dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.initializer.XavierNormal(self.fc1.weight)
        nn.initializer.XavierNormal(self.fc2.weight)
        nn.initializer.Normal(mean=0, std=1e-6)(self.fc1.bias)
        nn.initializer.Normal(mean=0, std=1e-6)(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Layer):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, hybrid_model, grid_size, hidden_size, dropout_rate,
                 img_size):
        super(Embeddings, self).__init__()

        img_size = (img_size, img_size)

        patch_size = (img_size[0] // 16 // grid_size[0],
                      img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] //
                                                           patch_size_real[1])

        self.hybrid_model = hybrid_model
        in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv2D(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size)
        self.position_embeddings = paddle.create_parameter(
            shape=[1, n_patches, hidden_size],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0))

        self.dropout = nn.Dropout(dropout_rate)
        self.hybrid = True

    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(
            x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = paddle.transpose(x, [0, 2, 1])

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Layer):
    def __init__(self, hidden_size, mlp_dim, dropout_rate, num_heads,
                 attention_dropout_rate, vis):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, epsilon=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, epsilon=1e-6)
        self.ffn = Mlp(hidden_size, mlp_dim, dropout_rate)
        self.attn = Attention(hidden_size, num_heads, attention_dropout_rate,
                              vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with paddle.no_grad():
            query_weight = np2th(weights[pjoin(
                ROOT, ATTENTION_Q, "kernel")]).reshape(
                    [self.hidden_size, self.hidden_size])
            key_weight = np2th(weights[pjoin(
                ROOT, ATTENTION_K, "kernel")]).reshape(
                    [self.hidden_size, self.hidden_size])
            value_weight = np2th(weights[pjoin(
                ROOT, ATTENTION_V, "kernel")]).reshape(
                    [self.hidden_size, self.hidden_size])
            out_weight = np2th(weights[pjoin(
                ROOT, ATTENTION_OUT, "kernel")]).reshape(
                    [self.hidden_size, self.hidden_size])

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q,
                                             "bias")]).reshape([-1])
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).reshape(
                [-1])
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V,
                                             "bias")]).reshape([-1])
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT,
                                           "bias")]).reshape([-1])

            self.attn.query.weight.set_value(query_weight)
            self.attn.key.weight.set_value(key_weight)
            self.attn.value.weight.set_value(value_weight)
            self.attn.out.weight.set_value(out_weight)
            self.attn.query.bias.set_value(query_bias)
            self.attn.key.bias.set_value(key_bias)
            self.attn.value.bias.set_value(value_bias)
            self.attn.out.bias.set_value(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")])
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")])
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).T
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).T

            self.ffn.fc1.weight.set_value(mlp_weight_0)
            self.ffn.fc2.weight.set_value(mlp_weight_1)
            self.ffn.fc1.bias.set_value(mlp_bias_0)
            self.ffn.fc2.bias.set_value(mlp_bias_1)

            self.attention_norm.weight.set_value(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.set_value(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.set_value(
                np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.set_value(
                np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Layer):
    def __init__(self, hidden_size, num_layers, mlp_dim, dropout_rate,
                 num_heads, attention_dropout_rate, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.LayerList()
        self.encoder_norm = nn.LayerNorm(hidden_size, epsilon=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, mlp_dim, dropout_rate, num_heads,
                          attention_dropout_rate, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Layer):
    def __init__(self, backbone, grid_size, hidden_size, dropout_rate,
                 num_layers, mlp_dim, num_heads, attention_dropout_rate,
                 img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(
            backbone, grid_size, hidden_size, dropout_rate, img_size=img_size)
        self.encoder = Encoder(hidden_size, num_layers, mlp_dim, dropout_rate,
                               num_heads, attention_dropout_rate, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(
            embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True, ):
        conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=not (use_batchnorm), )
        relu = nn.ReLU()

        bn = nn.BatchNorm2D(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True, ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm, )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm, )
        self.up = nn.UpsamplingBilinear2D(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = paddle.concat([x, skip], axis=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2D(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Layer):
    def __init__(self, hidden_size, decoder_channels, n_skip, skip_channels):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True, )
        decoder_channels = decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.n_skip = n_skip
        if n_skip != 0:
            skip_channels = skip_channels
            for i in range(4 - n_skip
                           ):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels,
                                            skip_channels)
        ]
        self.blocks = nn.LayerList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.shape  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.transpose([0, 2, 1])
        x = x.reshape([B, hidden, h, w])
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


@manager.MODELS.add_component
class TransUNet(nn.Layer):
    def __init__(self,
                 backbone,
                 classifier="seg",
                 decoder_channels=[256, 128, 64, 16],
                 hidden_size=768,
                 n_skip=3,
                 patches_grid=[14, 14],
                 pretrained_path=None,
                 skip_channels=[512, 256, 64, 16],
                 attention_dropout_rate=0.0,
                 dropout_rate=0.1,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 img_size=224,
                 num_classes=9,
                 zero_head=False,
                 vis=False):
        super(TransUNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = classifier
        self.transformer = Transformer(
            backbone, patches_grid, hidden_size, dropout_rate, num_layers,
            mlp_dim, num_heads, attention_dropout_rate, img_size, vis)
        self.decoder = DecoderCup(hidden_size, decoder_channels, n_skip,
                                  skip_channels)
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3, )
        if pretrained_path is not None:
            load_pretrained_model(self, pretrained_path)

    def forward(self, x):
        if self.training:
            x = paddle.squeeze(x, axis=2)
        else:
            x = paddle.squeeze(x, axis=0)
        x = x.tile([1, 3, 1, 1])
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        logits = paddle.unsqueeze(logits, axis=2)
        return [logits]


def export_weight_names(net):
    print(net.state_dict().keys())
    with open('paddle.txt', 'w') as f:
        for key in net.state_dict().keys():
            f.write(key + '\n')

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

# Adapted from https://github.com/facebookresearch/Mask2Former
#
# Original copyright info:

# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import param_init

from paddlepanseg.models.param_init import THLinearInitMixin
from paddlepanseg.utils import use_custom_op


def slow_ms_deform_attn(value, value_spatial_shapes, sampling_locations,
                        attention_weights):
    b, _, num_heads, depth = value.shape
    _, n_queries, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in value_spatial_shapes], axis=1)
    # Normalize [0, 1] -> [-1, 1]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    # Iterate along levels
    for i, (h, w) in enumerate(value_spatial_shapes):
        value_l_ = value_list[i].flatten(2).transpose((0, 2, 1)).reshape(
            (b * num_heads, depth, h, w))
        sampling_grid_l_ = sampling_grids[:, :, :, i].transpose(
            (0, 2, 1, 3, 4)).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_,
                                          sampling_grid_l_,
                                          mode='bilinear',
                                          padding_mode='zeros',
                                          align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (b, n_queries, num_heads, num_levels*num_points) -> (b, num_heads, n_queries, num_levels*num_points) -> (b, num_heads, 1, n_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose((0, 2, 1, 3, 4)).reshape(
        (b * num_heads, 1, n_queries, num_levels * num_points))
    output = (paddle.stack(sampling_value_list, axis=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(
                  (b, num_heads * depth, n_queries))
    return output.transpose((0, 2, 1))


class LinearWithFrozenBias(nn.Layer):

    def __init__(self,
                 in_features,
                 out_features,
                 bias_init_val,
                 weight_attr=None,
                 name=None):
        super().__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self.weight = self.create_parameter(shape=[in_features, out_features],
                                            attr=self._weight_attr,
                                            dtype=self._dtype,
                                            is_bias=False)
        bias_init_val = bias_init_val.flatten()
        bias_init_val = bias_init_val.astype(self._dtype)
        if bias_init_val.shape[0] != out_features:
            raise ValueError
        self.register_buffer('bias', bias_init_val)
        self.name = name

    def forward(self, input):
        out = paddle.matmul(input, self.weight)
        out = out + self.bias
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'in_features={}, out_features={}, dtype={}{}'.format(
            self.weight.shape[0], self.weight.shape[1], self._dtype, name_str)


class MSDeformAttn(nn.Layer, THLinearInitMixin):

    def __init__(self, embed_dim=256, num_levels=4, num_heads=8, num_points=4):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "`embed_dim` must be divisible by `num_heads`, but got {} and {}"
                .format(embed_dim, num_heads))

        self.im2col_step = 128

        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.init_weight()

    def init_weight(self):
        super().init_weight()
        param_init.constant_init(self.sampling_offsets.weight, value=0.)
        thetas = paddle.arange(
            self.num_heads, dtype='float32') * (2.0 * math.pi / self.num_heads)
        grid_init = paddle.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = paddle.tile(
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).reshape(
                (self.num_heads, 1, 1, 2)),
            (1, self.num_levels, self.num_points, 1))
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        paddle.nn.initializer.Assign(grid_init.flatten())(
            self.sampling_offsets.bias)
        self.sampling_offsets.bias.stop_gradient = True

        param_init.constant_init(self.attention_weights.weight, value=0.)
        param_init.constant_init(self.attention_weights.bias, value=0.)
        param_init.xavier_uniform(self.value_proj.weight)
        param_init.constant_init(self.value_proj.bias, value=0.)
        param_init.xavier_uniform(self.output_proj.weight)
        param_init.constant_init(self.output_proj.bias, value=0.)

    def forward(self,
                query,
                reference_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index,
                input_padding_mask=None):
        n, len_q, _ = query.shape
        n, len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] *
                input_spatial_shapes[:, 1]).sum() == len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = paddle.where(input_padding_mask[..., None],
                                 paddle.zeros_like(value), value)
        value = value.reshape(
            (n, len_in, self.num_heads, self.embed_dim // self.num_heads))
        sampling_offsets = self.sampling_offsets(query).reshape(
            (n, len_q, self.num_heads, self.num_levels, self.num_points, 2))
        attention_weights = self.attention_weights(query).reshape(
            (n, len_q, self.num_heads, self.num_levels * self.num_points))
        attention_weights = F.softmax(attention_weights, -1).reshape(
            (n, len_q, self.num_heads, self.num_levels, self.num_points))
        # n, len_q, num_heads, num_levels, num_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = paddle.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]],
                -1).astype('float32')
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead."
                .format(reference_points.shape[-1]))
        if paddle.is_compiled_with_cuda():
            # GPU
            with use_custom_op('ms_deform_attn') as msda:
                output = msda.ms_deform_attn(value, input_spatial_shapes,
                                             input_level_start_index,
                                             sampling_locations,
                                             attention_weights,
                                             self.im2col_step)
        else:
            # CPU
            output = slow_ms_deform_attn(value, input_spatial_shapes,
                                         sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

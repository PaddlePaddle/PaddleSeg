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

import math

import paddle
from paddleseg.cvlibs import param_init


def c2_xavier_fill(layer):
    param_init.kaiming_uniform(
        layer.weight, negative_slope=1, nonlinearity='leaky_relu')
    if layer.bias is not None:
        param_init.constant_init(layer.bias, value=0)


def c2_msra_fill(layer):
    n = layer._kernel_size[0] * layer._kernel_size[1] * layer._out_channels
    param_init.normal_init(layer.weight, std=math.sqrt(2. / n))
    if layer.bias is not None:
        param_init.constant_init(layer.bias, value=0)


def th_multihead_fill(layer, qkv_same_embed_dim=False):
    def _init_param_as_combined_linear_weight(p):
        bound = math.sqrt(6 / (3 * p.shape[0] + p.shape[1]))
        paddle.nn.initializer.Uniform(low=-bound, high=bound)(p)

    if qkv_same_embed_dim:
        _init_param_as_combined_linear_weight(layer.q_proj.weight)
        _init_param_as_combined_linear_weight(layer.k_proj.weight)
        _init_param_as_combined_linear_weight(layer.v_proj.weight)
        param_init.xavier_uniform(layer.out_proj.weight)
    else:
        for p in layer.parameters():
            if p.dim() > 1:
                param_init.xavier_uniform(p)


def th_linear_fill(layer):
    paddle.nn.initializer.KaimingUniform(
        negative_slope=math.sqrt(5), nonlinearity='leaky_relu')(layer.weight)
    if getattr(layer, 'bias', None) is not None:
        fan_in = layer.weight.shape[0]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        paddle.nn.initializer.Uniform(low=-bound, high=bound)(layer.bias)


class THLinearInitMixin(object):
    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, paddle.nn.Linear):
                th_linear_fill(layer)

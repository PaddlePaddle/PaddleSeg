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

# This implementation refers to: https://github.com/facebookresearch/segment-anything

import paddle
import paddle.nn as nn
from typing import Type


class MLPBlock(nn.Layer):
    def __init__(self,
                 embedding_dim: int,
                 mlp_dim: int,
                 act: Type[nn.Layer]=nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Layer):
    def __init__(self, num_channels: int, eps: float=1e-06) -> None:
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[num_channels],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=1.0))
        self.bias = paddle.create_parameter(
            shape=[num_channels],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0.0))
        self.eps = eps

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / paddle.sqrt(s + self.eps)
        x = self.weight[:, (None), (None)] * x + self.bias[:, (None), (None)]
        return x

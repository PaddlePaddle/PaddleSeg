# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

__all__ = ['to_2tuple', 'DropPath']


def to_2tuple(x):
    return tuple([x] * 2)


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

    def drop_path(self, x, drop_prob=0., training=False):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = paddle.to_tensor(1 - drop_prob)
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
        random_tensor = paddle.floor(random_tensor)  # binarize
        output = x.divide(keep_prob) * random_tensor
        return output

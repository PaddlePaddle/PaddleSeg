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

import paddle


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
    (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # new axis order
    axis_order = (1, 0) + tuple(range(2, len(tensor.shape)))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = paddle.transpose(tensor, perm=axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return paddle.flatten(transposed, start_axis=1, stop_axis=-1)


def class_weights(tensor):
    # normalize the input first
    tensor = paddle.nn.functional.softmax(tensor, axis=1)
    flattened = flatten(tensor)
    nominator = (1. - flattened).sum(-1)
    denominator = flattened.sum(-1)
    class_weights = nominator / denominator
    class_weights.stop_gradient = True

    return class_weights

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class CrossEntropyLoss(nn.Layer):
    """
    Implements the cross entropy loss function.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.EPS = 1e-5

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """
        if len(label.shape) != len(logit.shape):
            label = paddle.unsqueeze(label, 1)

        logit = paddle.transpose(logit, [0, 2, 3, 1])
        label = paddle.transpose(label, [0, 2, 3, 1])
        loss = F.softmax_with_cross_entropy(
            logit, label, ignore_index=self.ignore_index, axis=-1)

        mask = label != self.ignore_index
        mask = paddle.cast(mask, 'float32')
        loss = loss * mask
        avg_loss = paddle.mean(loss) / (paddle.mean(mask) + self.EPS)

        label.stop_gradient = True
        mask.stop_gradient = True
        return avg_loss

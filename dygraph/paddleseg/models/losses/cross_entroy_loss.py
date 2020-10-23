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
'''
@manager.LOSSES.add_component
class CrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Implements the cross entropy loss function.

    Args:
        weight (Tensor): Weight tensor, a manual rescaling weight given
            to each class and the shape is (C). It has the same dimensions as class
            number and the data type is float32, float64. Default ``'None'``.
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        reduction (str): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`size_average` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default ``'mean'``.

    """

    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.EPS = 1e-5
        if self.reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in cross_entropy_loss should be 'sum', 'mean' or"
                " 'none', but received %s, which is not allowed." %
                self.reduction)

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
        loss = paddle.nn.functional.cross_entropy(
            logit,
            label,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction)

        mask = label != self.ignore_index
        mask = paddle.cast(mask, 'float32')
        avg_loss = loss / (paddle.mean(mask) + self.EPS)

        label.stop_gradient = True
        mask.stop_gradient = True
        return avg_loss
'''


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

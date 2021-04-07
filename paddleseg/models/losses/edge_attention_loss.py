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
from paddleseg.models import losses


@manager.LOSSES.add_component
class EdgeAttentionLoss(nn.Layer):
    """
    Implements the cross entropy loss function. It only compute the edge part.

    Args:
        edge_threshold (float): The pixels greater edge_threshold as edges.
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, edge_threshold=0.8, ignore_index=255):
        super().__init__()
        self.edge_threshold = edge_threshold
        self.ignore_index = ignore_index
        self.EPS = 1e-10
        self.mean_mask = 1

    def forward(self, logits, label):
        """
        Forward computation.

        Args:
            logits (tuple|list): (seg_logit, edge_logit) Tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1. C =1 of edge_logit .
            label (Tensor): Label tensor, the data type is int64. Shape is (N, C), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, C, D1, D2,..., Dk), k >= 1.
        """
        seg_logit, edge_logit = logits[0], logits[1]
        if len(label.shape) != len(seg_logit.shape):
            label = paddle.unsqueeze(label, 1)
        if edge_logit.shape != label.shape:
            raise ValueError(
                'The shape of edge_logit should equal to the label, but they are {} != {}'
                .format(edge_logit.shape, label.shape))

        filler = paddle.ones_like(label) * self.ignore_index
        label = paddle.where(edge_logit > self.edge_threshold, label, filler)

        seg_logit = paddle.transpose(seg_logit, [0, 2, 3, 1])
        label = paddle.transpose(label, [0, 2, 3, 1])
        loss = F.softmax_with_cross_entropy(
            seg_logit, label, ignore_index=self.ignore_index, axis=-1)

        mask = label != self.ignore_index
        mask = paddle.cast(mask, 'float32')
        loss = loss * mask
        avg_loss = paddle.mean(loss) / (paddle.mean(mask) + self.EPS)
        if paddle.mean(mask) < self.mean_mask:
            self.mean_mask = paddle.mean(mask)

        label.stop_gradient = True
        mask.stop_gradient = True
        return avg_loss

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
class BootstrappedCrossEntropyLoss(nn.Layer):
    """
    Implements the cross entropy loss function.

    Args:
        min_K (int): the minimum number of pixels to be counted in loss computation.
        loss_th (float): the loss threshold. Only loss that is larger than the threshold
            would be calculated.
        weight (tuple|list, optional): The weight for different classes. Default: None.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default: 255.
    """

    def __init__(self, min_K, loss_th, weight=None, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.K = min_K
        self.threshold = loss_th
        if weight is not None:
            weight = paddle.to_tensor(weight, dtype='float32')
        self.weight = weight

    def forward(self, logit, label):

        n, c, h, w = logit.shape
        total_loss = 0.0
        if len(label.shape) != len(logit.shape):
            label = paddle.unsqueeze(label, 1)

        for i in range(n):
            x = paddle.unsqueeze(logit[i], 0)
            y = paddle.unsqueeze(label[i], 0)
            x = paddle.transpose(x, (0, 2, 3, 1))
            y = paddle.transpose(y, (0, 2, 3, 1))
            x = paddle.reshape(x, shape=(-1, c))
            y = paddle.reshape(y, shape=(-1, ))
            loss = F.cross_entropy(
                x,
                y,
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction="none")
            sorted_loss = paddle.sort(loss, descending=True)
            if sorted_loss[self.K] > self.threshold:
                new_indices = paddle.nonzero(sorted_loss > self.threshold)
                loss = paddle.gather(sorted_loss, new_indices)
            else:
                loss = sorted_loss[:self.K]

            total_loss += paddle.mean(loss)
        return total_loss / float(n)

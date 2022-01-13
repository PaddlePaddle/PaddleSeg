# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class KLLoss(nn.Layer):
    """
    The implementation of Kullback-Leibler divergence Loss.
    Refer to https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        temperature (float): the coefficient of kl_loss.
    """

    def __init__(self, ignore_index=255, temperature=1):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature

        self.kl_loss = nn.KLDivLoss(reduction="none")
        self.EPS = 1e-8

    def forward(self, logit_1, logit_2, label=None):
        """
        Calculate the KL loss. If the label is not None, it considers the
        ignore_index in label and calculates the masked loss.

        Args:
            logit_1 (Tensor): Logit tensor, the data type is float32 or float64.
                The shape is (N, C), where C is number of classes, and if shape is
                more than 2D, this is (N, C, D1, D2,..., Dk), k >= 1.
            logit_2 (Tensor): Logit tensor, the data type is float32 or float64.
                The shape of logit_2 and logit_1 are the same.
            label (Tensor, optional): Label tensor, the data type is int64.
                The shape is (N), where each value is 0 <= label[i] <= C-1, and
                if shape is more than 2D, this is (N, D1, D2,..., Dk), k >= 1.
        Returns:
            (Tensor): The average loss.
        """
        if logit_1.shape != logit_2.shape:
            raise ValueError(
                'The shape of logit_1 = {} must be the same as the shape of logit_2 = {}.'
                .format(logit_1.shape, logit_2.shape))

        logit_1 = F.log_softmax(logit_1 / self.temperature, axis=1)
        logit_2 = F.softmax(logit_2 / self.temperature, axis=1)
        loss = self.kl_loss(logit_1, logit_2)
        loss = loss * self.temperature * self.temperature

        if label is None:
            avg_loss = paddle.mean(loss)
        else:
            mask = label != self.ignore_index
            mask = paddle.cast(mask, 'float32')
            mask = paddle.unsqueeze(mask, axis=1)
            label.stop_gradient = True
            mask.stop_gradient = True

            loss = loss * mask
            avg_loss = paddle.mean(loss) / (paddle.mean(mask) + self.EPS)
        return avg_loss

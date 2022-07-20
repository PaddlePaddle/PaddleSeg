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

import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F
from scipy.ndimage import shift

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class RelaxBoundaryLoss(nn.Layer):
    """
    Implements the ohem cross entropy loss function.

    Args:
        border (int, optional): The value of border to relax. Default: 1.
        calculate_weights (bool, optional): Whether to calculate weights for every classes. Default: False.
        upper_bound (float, optional): The upper bound of weights if calculating weights for every classes. Default: 1.0.
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default: 255.
    """

    def __init__(self,
                 border=1,
                 calculate_weights=False,
                 upper_bound=1.0,
                 ignore_index=255):
        super(RelaxBoundaryLoss, self).__init__()
        self.border = border
        self.calculate_weights = calculate_weights
        self.upper_bound = upper_bound
        self.ignore_index = ignore_index
        self.EPS = 1e-5

    def relax_onehot(self, label, num_classes):
        # pad label, and let ignore_index as num_classes
        if len(label.shape) == 3:
            label = label.unsqueeze(1)
        h, w = label.shape[-2], label.shape[-1]
        label = F.pad(label, [self.border] * 4, value=num_classes)
        label = label.squeeze(1)
        ignore_mask = (label == self.ignore_index).astype('int64')
        label = label * (1 - ignore_mask) + num_classes * ignore_mask

        onehot = 0
        for i in range(-self.border, self.border + 1):
            for j in range(-self.border, self.border + 1):
                h_start, h_end = 1 + i, h + 1 + i
                w_start, w_end = 1 + j, w + 1 + j
                label_ = label[:, h_start:h_end, w_start:w_end]
                onehot_ = F.one_hot(label_, num_classes + 1)
                onehot += onehot_
        onehot = (onehot > 0).astype('int64')
        onehot = paddle.transpose(onehot, (0, 3, 1, 2))

        return onehot

    def calculate_weights(self, label):
        hist = paddle.sum(label, axis=(1, 2)) * 1.0 / label.sum()
        hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1

    def custom_nll(self,
                   logit,
                   label,
                   class_weights=None,
                   border_weights=None,
                   ignore_mask=None):
        soft = F.softmax(logit, axis=1)
        # calculate the valid soft where label is 1.
        soft_label = ((soft * label[:, :-1, :, :]).sum(
            1, keepdim=True)) * (label[:, :-1, :, :].astype('float32'))
        soft = soft * (1 - label[:, :-1, :, :]) + soft_label
        logsoft = paddle.log(soft)
        if class_weights is not None:
            logsoft = class_weights.unsqueeze((0, 2, 3))
        logsoft = label[:, :-1, :, :] * logsoft
        logsoft = logsoft.sum(1)
        # border loss is divided equally
        logsoft = -1 / border_weights * logsoft * (1. - ignore_mask)
        n, _, h, w = label.shape
        logsoft = logsoft.sum() / (n * h * w - ignore_mask.sum() + 1)
        return logsoft

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
        n, c, h, w = logit.shape
        label.stop_gradient = True
        label = self.relax_onehot(label, c)
        weights = label[:, :-1, :, :].sum(1).astype('float32')
        ignore_mask = (weights == 0).astype('float32')
        # border is greater than 1, other is 1
        border_weights = weights + ignore_mask

        loss = 0
        class_weights = None
        for i in range(n):
            if self.calculate_weights:
                class_weights = self.calculate_weights(label[i])
            loss = loss + self.custom_nll(
                logit[i].unsqueeze(0),
                label[i].unsqueeze(0),
                class_weights=class_weights,
                border_weights=border_weights,
                ignore_mask=ignore_mask[i])
        return loss

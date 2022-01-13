# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
class MRSD(nn.Layer):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logit, label, mask=None):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64.
            label (Tensor): Label tensor, the data type is float32, float64. The shape should equal to logit.
            mask (Tensor, optional): The mask where the loss valid. Default： None.
        """
        if len(label.shape) == 3:
            label = label.unsqueeze(1)
        sd = paddle.square(logit - label)
        loss = paddle.sqrt(sd + self.eps)
        if mask is not None:
            mask = mask.astype('float32')
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + self.eps)
            mask.stop_gradient = True
        else:
            loss = loss.mean()

        return loss

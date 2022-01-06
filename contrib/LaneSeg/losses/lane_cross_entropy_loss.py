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
class LaneCrossEntropyLoss(nn.Layer):
    def __init__(self, ignore_index=255, weights=None, data_format='NCHW'):
        super(LaneCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weights = weights

    def forward(self, logit, label, semantic_weights=None):
        temp = F.log_softmax(logit, axis=1)
        loss_func = nn.NLLLoss(
            ignore_index=self.ignore_index,
            weight=paddle.to_tensor(self.weights))
        loss = loss_func(temp, label)
        return loss

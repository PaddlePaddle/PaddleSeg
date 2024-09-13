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
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class DistillationLoss(nn.Layer):
    """
    For STFPM network
    """

    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.EPS = 1e-10

    def forward(self, inputs, label):
        s_feat, t_feat = inputs
        loss = []
        for i in range(len(t_feat)):
            t_feat[i] = F.normalize(t_feat[i], axis=1)
            s_feat[i] = F.normalize(s_feat[i], axis=1)
            loss.append(paddle.sum((t_feat[i] - s_feat[i])**2, 1).mean())
        return sum(loss)

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

import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class KLLoss(nn.Layer):
    """
    The Kullback-Leibler divergence Loss implement.

    Code referenced from:

    https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py



    Compute the knowledge-distillation (KD) loss given outputs, labels.

    "Hyperparameters": temperature and alpha

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        temperature (float): the coefficient of kl_loss.
    """

    def __init__(self, ignore_index=255, temperature=1):
        super(KLLoss, self).__init__()
        self.ignore_index = ignore_index
        self.kl_loss = nn.KLDivLoss(reduction="mean")
        self.temperature = temperature

    def forward(self, logit, label):
        logit = F.log_softmax(logit / self.temperature, axis=1)
        label = F.softmax(label / self.temperature, axis=1)
        return self.kl_loss(logit, label) * self.temperature * self.temperature

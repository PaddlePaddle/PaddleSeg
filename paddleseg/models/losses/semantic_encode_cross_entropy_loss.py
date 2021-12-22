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
class SECrossEntropyLoss(nn.Layer):
    """
    The Semantic Encoding Loss implementation based on PaddlePaddle.

    """
    def __init__(self, *args, **kwargs):
        super(SECrossEntropyLoss, self).__init__()

    def forward(self, logit, label):
        if logit.ndim == 4:
            logit = logit.squeeze(2).squeeze(3)
        assert logit.ndim == 2, "The shape of logit should be [N, C, 1, 1] or [N, C], but the logit dim is  {}.".format(
            logit.ndim)

        batch_size, num_classes = paddle.shape(logit)
        se_label = paddle.zeros([batch_size, num_classes])
        for i in range(batch_size):
            hist = paddle.histogram(label[i],
                                    bins=num_classes,
                                    min=0,
                                    max=num_classes - 1)
            hist = hist.astype('float32') / hist.sum().astype('float32')
            se_label[i] = (hist > 0).astype('float32')
        loss = F.binary_cross_entropy_with_logits(logit, se_label)
        return loss

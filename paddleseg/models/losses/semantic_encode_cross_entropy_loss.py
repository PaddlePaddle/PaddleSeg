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
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class SECrossEntropyLoss(nn.Layer):
    """
    The Semantic Encoding Loss implementation based on PaddlePaddle.

    Args:
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default: 255.
    """
    def __init__(self, num_classes, ignore_index=255):
        super(SECrossEntropyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logit, label):
        if logit.ndim == 4:
            logit_ = logit.sequeeze(2).sequeeze(3)
            assert logit_.ndim == 2, "The shape of logit should be [N, C, 1, 1] or [N, C], but got {}.".format(
                logit.ndim)
            logit = logit_
        assert logit.ndim == 2, "The dimension of logit should be 2, but got {}.".format(
            logit.ndim)
        batch_size = paddle.shape(label)[0]
        se_label = paddle.zeros([batch_size, self.num_classes])
        for i in range(batch_size):
            hist = paddle.histogram(label[i],
                                    bins=self.num_classes,
                                    min=0,
                                    max=self.num_classes - 1)
            hist = hist.astype('float32') / hist.sum().astype('float32')
            se_label[i] = (hist > 0).astype('float32')
        loss = F.binary_cross_entropy_with_logits(logit, se_label)
        return loss

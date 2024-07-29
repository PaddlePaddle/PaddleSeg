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

_IS_NPU = "npu" in paddle.get_device()
_IS_MLU = "mlu" in paddle.get_device()


@manager.LOSSES.add_component
class OhemCrossEntropyLoss(nn.Layer):
    """
    Implements the ohem cross entropy loss function.

    Args:
        thresh (float, optional): The threshold of ohem. Default: 0.7.
        min_kept (int, optional): The min number to keep in loss computation. Default: 10000.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
    """

    def __init__(self,
                 thresh=0.7,
                 min_kept=10000,
                 ignore_index=255,
                 weight=None):
        super(OhemCrossEntropyLoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.EPS = 1e-5
        if weight is not None:
            self.weight = paddle.to_tensor(weight, dtype='float32')
        else:
            self.weight = None

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
        if self.weight is not None and logit.shape[1] != len(self.weight):
            raise ValueError(
                'The number of weights = {} must be the same as the number of classes = {}.'
                .format(len(self.weight), logit.shape[1]))

        if len(label.shape) != len(logit.shape):
            label = paddle.unsqueeze(label, 1)

        # get the label after ohem
        n, c, h, w = logit.shape
        label = label.reshape((-1, )).astype('int64')
        valid_mask = (label != self.ignore_index).astype('int64')
        num_valid = valid_mask.sum()
        label = label * valid_mask

        prob = F.softmax(logit, axis=1)
        prob = prob.transpose((1, 0, 2, 3)).reshape((c, -1))

        if self.min_kept < num_valid and num_valid > 0:
            # let the value which ignored greater than 1
            prob = prob + (1 - valid_mask).astype(prob.dtype)

            # get the prob of relevant label
            label_onehot = F.one_hot(label, c)
            label_onehot = label_onehot.transpose((1, 0))
            prob = prob * label_onehot
            prob = paddle.sum(prob, axis=0)

            threshold = self.thresh
            if self.min_kept > 0:
                index = prob.argsort()
                if hasattr(paddle.Tensor, "contiguous"):
                    threshold_index = index[min(len(index), self.min_kept) -
                                            1].contiguous()
                else:
                    threshold_index = index[min(len(index), self.min_kept) - 1]
                threshold_index = int(threshold_index)
                if prob[threshold_index] > self.thresh:
                    threshold = prob[threshold_index]
                kept_mask = (prob < threshold).astype('int64')
                label = label * kept_mask
                valid_mask = valid_mask * kept_mask

        # make the invalid region as ignore
        label = label + (1 - valid_mask) * self.ignore_index

        label = label.reshape((n, 1, h, w))
        valid_mask = valid_mask.reshape((n, 1, h, w)).astype('float32')

        if _IS_NPU:
            logit = F.log_softmax(logit, axis=1)
            loss = F.nll_loss(logit,
                              label.squeeze(1),
                              weight=self.weight,
                              ignore_index=self.ignore_index,
                              reduction='none')
            loss = loss.unsqueeze(1)
        elif _IS_MLU:
            # mlu kernel does not deal with th parameter ignore_index
            # so here manual solve it
            logit_trans = logit.transpose((0, 2, 3, 1)).flatten(0, 2)
            mask = label != self.ignore_index
            label = paddle.where(mask, label, paddle.zeros_like(label))
            label_trans = label.transpose((0, 2, 3, 1)).flatten(0, 2)
            loss = F.cross_entropy(logit_trans,
                                   label_trans,
                                   weight=self.weight,
                                   reduction='none',
                                   axis=-1)
            loss = loss.reshape([n, h, w, 1]).transpose([0, 3, 1, 2])
            loss = loss * mask.astype("float32")
        else:
            loss = F.cross_entropy(logit,
                                   label,
                                   weight=self.weight,
                                   ignore_index=self.ignore_index,
                                   reduction='none',
                                   axis=1)
        loss = loss * valid_mask
        avg_loss = paddle.mean(loss) / (paddle.mean(valid_mask) + self.EPS)

        label.stop_gradient = True
        valid_mask.stop_gradient = True
        return avg_loss

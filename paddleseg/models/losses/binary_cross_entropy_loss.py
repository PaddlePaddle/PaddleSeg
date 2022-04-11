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
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class BCELoss(nn.Layer):
    r"""
    This operator combines the sigmoid layer and the :ref:`api_nn_loss_BCELoss` layer.
    Also, we can see it as the combine of ``sigmoid_cross_entropy_with_logits``
    layer and some reduce operations.
    This measures the element-wise probability error in classification tasks
    in which each class is independent.
    This can be thought of as predicting labels for a data-point, where labels
    are not mutually exclusive. For example, a news article can be about
    politics, technology or sports at the same time or none of these.
    First this operator calculate loss function as follows:
    .. math::
           Out = -Labels * \\log(\\sigma(Logit)) - (1 - Labels) * \\log(1 - \\sigma(Logit))
    We know that :math:`\\sigma(Logit) = \\frac{1}{1 + \\e^{-Logit}}`. By substituting this we get:
    .. math::
           Out = Logit - Logit * Labels + \\log(1 + \\e^{-Logit})
    For stability and to prevent overflow of :math:`\\e^{-Logit}` when Logit < 0,
    we reformulate the loss as follows:
    .. math::
           Out = \\max(Logit, 0) - Logit * Labels + \\log(1 + \\e^{-\|Logit\|})
    Then, if ``weight`` or ``pos_weight`` is not None, this operator multiply the
    weight tensor on the loss `Out`. The ``weight`` tensor will attach different
    weight on every items in the batch. The ``pos_weight`` will attach different
    weight on the positive label of each class.
    Finally, this operator applies reduce operation on the loss.
    If :attr:`reduction` set to ``'none'``, the operator will return the original loss `Out`.
    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is :math:`Out = MEAN(Out)`.
    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is :math:`Out = SUM(Out)`.
    Note that the target labels ``label`` should be numbers between 0 and 1.
    Args:
        weight (Tensor | str, optional): A manual rescaling weight given to the loss of each
            batch element. If given, it has to be a 1D Tensor whose size is `[N, ]`,
            The data type is float32, float64. If type is str, it should equal to 'dynamic'.
            It will compute weight dynamically in every step.
            Default is ``'None'``.
        pos_weight (float|str, optional): A weight of positive examples. If type is str,
            it should equal to 'dynamic'. It will compute weight dynamically in every step.
            Default is ``'None'``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        edge_label (bool, optional): Whether to use edge label. Default: False
    Shapes:
        logit (Tensor): The input predications tensor. 2-D tensor with shape: [N, *],
            N is batch_size, `*` means number of additional dimensions. The ``logit``
            is usually the output of Linear layer. Available dtype is float32, float64.
        label (Tensor): The target labels tensor. 2-D tensor with the same shape as
            ``logit``. The target labels which values should be numbers between 0 and 1.
            Available dtype is float32, float64.
    Returns:
        A callable object of BCEWithLogitsLoss.
    Examples:
        .. code-block:: python
            import paddle
            paddle.disable_static()
            logit = paddle.to_tensor([5.0, 1.0, 3.0], dtype="float32")
            label = paddle.to_tensor([1.0, 0.0, 1.0], dtype="float32")
            bce_logit_loss = paddle.nn.BCEWithLogitsLoss()
            output = bce_logit_loss(logit, label)
            print(output.numpy())  # [0.45618808]
    """

    def __init__(self,
                 weight=None,
                 pos_weight=None,
                 ignore_index=255,
                 edge_label=False):
        super().__init__()
        self.weight = weight
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index
        self.edge_label = edge_label
        self.EPS = 1e-10

        if self.weight is not None:
            if isinstance(self.weight, str):
                if self.weight != 'dynamic':
                    raise ValueError(
                        "if type of `weight` is str, it should equal to 'dynamic', but it is {}"
                        .format(self.weight))
            elif not isinstance(self.weight, paddle.Tensor):
                raise TypeError(
                    'The type of `weight` is wrong, it should be Tensor or str, but it is {}'
                    .format(type(self.weight)))

        if self.pos_weight is not None:
            if isinstance(self.pos_weight, str):
                if self.pos_weight != 'dynamic':
                    raise ValueError(
                        "if type of `pos_weight` is str, it should equal to 'dynamic', but it is {}"
                        .format(self.pos_weight))
            elif isinstance(self.pos_weight, float):
                self.pos_weight = paddle.to_tensor(
                    self.pos_weight, dtype='float32')
            else:
                raise TypeError(
                    'The type of `pos_weight` is wrong, it should be float or str, but it is {}'
                    .format(type(self.pos_weight)))

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N, C), where each
                value is 0 or 1, and if shape is more than 2D, this is
                (N, C, D1, D2,..., Dk), k >= 1.
        """
        if len(label.shape) != len(logit.shape):
            label = paddle.unsqueeze(label, 1)
        mask = (label != self.ignore_index)
        mask = paddle.cast(mask, 'float32')
        # label.shape should equal to the logit.shape
        if label.shape[1] != logit.shape[1]:
            label = label.squeeze(1)
            label = F.one_hot(label, logit.shape[1])
            label = label.transpose((0, 3, 1, 2))
        if isinstance(self.weight, str):
            pos_index = (label == 1)
            neg_index = (label == 0)
            pos_num = paddle.sum(pos_index.astype('float32'))
            neg_num = paddle.sum(neg_index.astype('float32'))
            sum_num = pos_num + neg_num
            weight_pos = 2 * neg_num / (sum_num + self.EPS)
            weight_neg = 2 * pos_num / (sum_num + self.EPS)
            weight = weight_pos * label + weight_neg * (1 - label)
        else:
            weight = self.weight
        if isinstance(self.pos_weight, str):
            pos_index = (label == 1)
            neg_index = (label == 0)
            pos_num = paddle.sum(pos_index.astype('float32'))
            neg_num = paddle.sum(neg_index.astype('float32'))
            sum_num = pos_num + neg_num
            pos_weight = 2 * neg_num / (sum_num + self.EPS)
        else:
            pos_weight = self.pos_weight
        label = label.astype('float32')
        loss = paddle.nn.functional.binary_cross_entropy_with_logits(
            logit,
            label,
            weight=weight,
            reduction='none',
            pos_weight=pos_weight)
        loss = loss * mask
        loss = paddle.mean(loss) / (paddle.mean(mask) + self.EPS)
        label.stop_gradient = True
        mask.stop_gradient = True

        return loss

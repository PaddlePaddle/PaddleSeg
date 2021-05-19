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
class L1Loss(nn.MSELoss):
    r"""
    This interface is used to construct a callable object of the ``L1Loss`` class.
    The L1Loss layer calculates the L1 Loss of ``input`` and ``label`` as follows.
     If `reduction` set to ``'none'``, the loss is:
    .. math::
        Out = \lvert input - label\rvert
    If `reduction` set to ``'mean'``, the loss is:
    .. math::
        Out = MEAN(\lvert input - label\rvert)
    If `reduction` set to ``'sum'``, the loss is:
    .. math::
        Out = SUM(\lvert input - label\rvert)

    Args:
        reduction (str, optional): Indicate the reduction to apply to the loss,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'none'``, the unreduced loss is returned;
            If `reduction` is ``'mean'``, the reduced mean loss is returned.
            If `reduction` is ``'sum'``, the reduced sum loss is returned.
            Default is ``'mean'``.
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default: 255.
    Shape:
        input (Tensor): The input tensor. The shapes is [N, *], where N is batch size and `*` means any number of additional dimensions. It's data type should be float32, float64, int32, int64.
        label (Tensor): label. The shapes is [N, *], same shape as ``input`` . It's data type should be float32, float64, int32, int64.
        output (Tensor): The L1 Loss of ``input`` and ``label``.
            If `reduction` is ``'none'``, the shape of output loss is [N, *], the same as ``input`` .
            If `reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [1].
    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            input_data = np.array([[1.5, 0.8], [0.2, 1.3]]).astype("float32")
            label_data = np.array([[1.7, 1], [0.4, 0.5]]).astype("float32")
            input = paddle.to_tensor(input_data)
            label = paddle.to_tensor(label_data)
            l1_loss = paddle.nn.L1Loss()
            output = l1_loss(input, label)
            print(output.numpy())
            # [0.35]
            l1_loss = paddle.nn.L1Loss(reduction='sum')
            output = l1_loss(input, label)
            print(output.numpy())
            # [1.4]
            l1_loss = paddle.nn.L1Loss(reduction='none')
            output = l1_loss(input, label)
            print(output)
            # [[0.20000005 0.19999999]
            # [0.2        0.79999995]]
    """

    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__(reduction=reduction)

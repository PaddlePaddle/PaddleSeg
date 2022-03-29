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
class MSELoss(nn.MSELoss):
    r"""
    **Mean Square Error Loss**
    Computes the mean square error (squared L2 norm) of given input and label.
    If :attr:`reduction` is set to ``'none'``, loss is calculated as:
    .. math::
        Out = (input - label)^2
    If :attr:`reduction` is set to ``'mean'``, loss is calculated as:
    .. math::
        Out = \operatorname{mean}((input - label)^2)
    If :attr:`reduction` is set to ``'sum'``, loss is calculated as:
    .. math::
        Out = \operatorname{sum}((input - label)^2)
    where `input` and `label` are `float32` tensors of same shape.

    Args:
        reduction (string, optional): The reduction method for the output,
            could be 'none' | 'mean' | 'sum'.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned.
            If :attr:`size_average` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default: 255.
    Shape:
        input (Tensor): Input tensor, the data type is float32 or float64
        label (Tensor): Label tensor, the data type is float32 or float64
        output (Tensor): output tensor storing the MSE loss of input and label, the data type is same as input.
    Examples:
        .. code-block:: python
            import numpy as np
            import paddle
            input_data = np.array([1.5]).astype("float32")
            label_data = np.array([1.7]).astype("float32")
            mse_loss = paddle.nn.loss.MSELoss()
            input = paddle.to_tensor(input_data)
            label = paddle.to_tensor(label_data)
            output = mse_loss(input, label)
            print(output)
            # [0.04000002]
    """

    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__(reduction=reduction)

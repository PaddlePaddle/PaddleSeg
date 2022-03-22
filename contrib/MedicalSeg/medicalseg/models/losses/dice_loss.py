# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from medicalseg.models.losses import flatten
from medicalseg.cvlibs import manager


@manager.LOSSES.add_component
class DiceLoss(nn.Layer):
    """
    Implements the dice loss function.

    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        smooth (float32): laplace smoothing,
            to smooth dice loss and accelerate convergence. following:
            https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895
    """

    def __init__(self, sigmoid_norm=True, weight=None):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.eps = 1e-5
        if sigmoid_norm:
            self.norm = nn.Sigmoid()
        else:
            self.norm = nn.Softmax(axis=1)

    def compute_per_channel_dice(self, input, target, epsilon=1e-6,
                                 weight=None):
        """
        Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
        Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

        Args:
            input (torch.Tensor): NxCxSpatial input tensor
            target (torch.Tensor): NxCxSpatial target tensor
            epsilon (float): prevents division by zero
            weight (torch.Tensor): Cx1 tensor of weight per channel/class
        """

        # input and target shapes must match
        assert input.shape == target.shape, "'input' and 'target' must have the same shape but input is {} and target is {}".format(
            input.shape, target.shape)

        input = flatten(input)  # C, N*D*H*W
        target = flatten(target)
        target = paddle.cast(target, "float32")

        # compute per channel Dice Coefficient
        intersect = (input * target).sum(-1)  # sum at the spatial dimension
        if weight is not None:
            intersect = weight * intersect  # give different class different weight

        # Use standard dice: (input + target).sum(-1) or V-Net extension: (input^2 + target^2).sum(-1)
        denominator = (input * input).sum(-1) + (target * target).sum(-1)

        return 2 * (intersect / paddle.clip(denominator, min=epsilon))

    def forward(self, logits, labels):
        """
        logits: tensor of [B, C, D, H, W]
        labels: tensor of shape [B, D, H, W]
        """
        assert "int" in str(labels.dtype), print(
            "The label should be int but got {}".format(type(labels)))
        if len(logits.shape) == 4:
            logits = logits.unsqueeze(0)

        labels_one_hot = F.one_hot(
            labels, num_classes=logits.shape[1])  # [B, D, H, W, C]
        labels_one_hot = paddle.transpose(labels_one_hot,
                                          [0, 4, 1, 2, 3])  # [B, C, D, H, W]

        labels_one_hot = paddle.cast(labels_one_hot, dtype='float32')

        logits = self.norm(logits)  # softmax to sigmoid

        per_channel_dice = self.compute_per_channel_dice(
            logits, labels_one_hot, weight=self.weight)

        dice_loss = (1. - paddle.mean(per_channel_dice))
        per_channel_dice = per_channel_dice.detach().cpu(
        ).numpy()  # vnet variant dice

        return dice_loss, per_channel_dice

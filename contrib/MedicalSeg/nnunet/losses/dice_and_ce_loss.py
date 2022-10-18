# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/MIC-DKFZ/nnUNet

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F

from medicalseg.cvlibs import manager


@manager.LOSSES.add_component
class DC_and_CE_loss(nn.Layer):
    def __init__(self,
                 batch_dice=True,
                 smooth=1e-5,
                 do_bg=False,
                 ce_kwargs={},
                 reduction="sum",
                 square_dice=False,
                 weight_ce=1,
                 weight_dice=1,
                 log_dice=False,
                 ignore_label=None):
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.reduction = reduction
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        self.dc = SoftDiceLoss(
            apply_softmax=True,
            batch_dice=batch_dice,
            smooth=smooth,
            do_bg=do_bg)

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc = self.dc(net_output, target,
                     loss_mask=mask) if self.weight_dice != 0 else 0
        dice_coef = dc
        dc_loss = -dc.mean()
        if self.log_dice:
            dc_loss = -paddle.log(-dc_loss)

        ce_loss = self.ce(
            net_output,
            target[:, 0].astype('float32')) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.reduction == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError(
                "reduction type not support, only support 'sum'.")
        return result, dice_coef.numpy()


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: paddle.Tensor,
                target: paddle.Tensor) -> paddle.Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        input = input.transpose([0, * [i for i in range(2, input.ndim)], 1])
        return super().forward(input, target.astype('int64'))


class SoftDiceLoss(nn.Layer):
    def __init__(self,
                 apply_softmax=True,
                 batch_dice=False,
                 do_bg=True,
                 smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_softmax = apply_softmax
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_softmax:
            x = F.softmax(x, axis=1)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        # dc = dc.mean()

        # return -dc
        return dc


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    shp_x = net_output.shape
    shp_y = gt.shape

    with paddle.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.reshape((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.astype('int64')[:, 0, ...]
            y_onehot = paddle.nn.functional.one_hot(
                gt, num_classes=net_output.shape[1])
            y_onehot = y_onehot.transpose([
                0, y_onehot.ndim - 1,
                * [i for i in range(1, y_onehot.ndim - 1)]
            ])

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = paddle.stack(
            tuple(x_i * mask[:, 0] for x_i in paddle.unbind(
                tp, axis=1)),
            axis=1)
        fp = paddle.stack(
            tuple(x_i * mask[:, 0] for x_i in paddle.unbind(
                fp, axis=1)),
            axis=1)
        fn = paddle.stack(
            tuple(x_i * mask[:, 0] for x_i in paddle.unbind(
                fn, axis=1)),
            axis=1)
        tn = paddle.stack(
            tuple(x_i * mask[:, 0] for x_i in paddle.unbind(
                tn, axis=1)),
            axis=1)
    if square:
        tp = tp**2
        fp = fp**2
        fn = fn**2
        tn = tn**2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)
    return tp, fp, fn, tn


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

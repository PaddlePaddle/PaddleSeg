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
"""Lovasz-Softmax and Jaccard hinge loss in PaddlePaddle"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class LovaszSoftmaxLoss(nn.Layer):
    """
    Multi-class Lovasz-Softmax loss.

    Args:
        ignore_index (int64): Specifies a target value that is ignored and does not contribute to the input gradient. Default ``255``.
        classes (str|list): 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """

    def __init__(self, ignore_index=255, classes='present'):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index
        self.classes = classes

    def forward(self, logits, labels):
        r"""
        Forward computation.

        Args:
            logits (Tensor): Shape is [N, C, H, W], logits at each prediction (between -\infty and +\infty).
            labels (Tensor): Shape is [N, 1, H, W] or [N, H, W], ground truth labels (between 0 and C - 1).
        """
        probas = F.softmax(logits, axis=1)
        vprobas, vlabels = flatten_probas(probas, labels, self.ignore_index)
        loss = lovasz_softmax_flat(vprobas, vlabels, classes=self.classes)
        return loss


@manager.LOSSES.add_component
class LovaszHingeLoss(nn.Layer):
    """
    Binary Lovasz hinge loss.

    Args:
        ignore_index (int64): Specifies a target value that is ignored and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, ignore_index=255):
        super(LovaszHingeLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        r"""
        Forward computation.

        Args:
            logits (Tensor): Shape is [N, 1, H, W] or [N, 2, H, W], logits at each pixel (between -\infty and +\infty).
            labels (Tensor): Shape is [N, 1, H, W] or [N, H, W], binary ground truth masks (0 or 1).
        """
        if logits.shape[1] == 2:
            logits = binary_channel_to_unary(logits)
        loss = lovasz_hinge_flat(
            *flatten_binary_scores(logits, labels, self.ignore_index))
        return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors.
    See Alg. 1 in paper.
    """
    gts = paddle.sum(gt_sorted)
    p = len(gt_sorted)

    intersection = gts - paddle.cumsum(gt_sorted, axis=0)
    union = gts + paddle.cumsum(1 - gt_sorted, axis=0)
    jaccard = 1.0 - intersection.cast('float32') / union.cast('float32')

    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def binary_channel_to_unary(logits, eps=1e-9):
    """
    Converts binary channel logits to unary channel logits for lovasz hinge loss.
    """
    probas = F.softmax(logits, axis=1)
    probas = probas[:, 1, :, :]
    logits = paddle.log(probas + eps / (1 - probas + eps))
    logits = logits.unsqueeze(1)
    return logits


def lovasz_hinge_flat(logits, labels):
    r"""
    Binary Lovasz hinge loss.

    Args:
        logits (Tensor): Shape is [P], logits at each prediction (between -\infty and +\infty).
        labels (Tensor): Shape is [P], binary ground truth labels (0 or 1).
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels - 1.
    signs.stop_gradient = True
    errors = 1. - logits * signs
    errors_sorted, perm = paddle._C_ops.argsort(errors, 'axis', 0, 'descending',
                                                True)
    errors_sorted.stop_gradient = False
    gt_sorted = paddle.gather(labels, perm)
    grad = lovasz_grad(gt_sorted)
    grad.stop_gradient = True
    loss = paddle.sum(F.relu(errors_sorted) * grad)
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case).
    Remove labels according to 'ignore'.
    """
    scores = paddle.reshape(scores, [-1])
    labels = paddle.reshape(labels, [-1])
    labels.stop_gradient = True
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    valid_mask = paddle.reshape(valid, (-1, 1))
    indexs = paddle.nonzero(valid_mask)
    indexs.stop_gradient = True
    vscores = paddle.gather(scores, indexs[:, 0])
    vlabels = paddle.gather(labels, indexs[:, 0])
    return vscores, vlabels


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss.

    Args:
        probas (Tensor): Shape is [P, C], class probabilities at each prediction (between 0 and 1).
        labels (Tensor): Shape is [P], ground truth labels (between 0 and C - 1).
        classes (str|list): 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.shape[1]
    losses = []
    classes_to_sum = list(range(C)) if classes in ['all', 'present'
                                                   ] else classes
    for c in classes_to_sum:
        fg = paddle.cast(labels == c, probas.dtype)  # foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        fg.stop_gradient = True
        if C == 1:
            if len(classes_to_sum) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = paddle.abs(fg - class_pred)
        errors_sorted, perm = paddle._C_ops.argsort(errors, 'axis', 0,
                                                    'descending', True)
        errors_sorted.stop_gradient = False

        fg_sorted = paddle.gather(fg, perm)
        fg_sorted.stop_gradient = True

        grad = lovasz_grad(fg_sorted)
        grad.stop_gradient = True
        loss = paddle.sum(errors_sorted * grad)
        losses.append(loss)

    if len(classes_to_sum) == 1:
        return losses[0]

    losses_tensor = paddle.stack(losses)
    mean_loss = paddle.mean(losses_tensor)
    return mean_loss


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch.
    """
    if len(probas.shape) == 3:
        probas = paddle.unsqueeze(probas, axis=1)
    C = probas.shape[1]
    probas = paddle.transpose(probas, [0, 2, 3, 1])
    probas = paddle.reshape(probas, [-1, C])
    labels = paddle.reshape(labels, [-1])
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    valid_mask = paddle.reshape(valid, [-1, 1])
    indexs = paddle.nonzero(valid_mask)
    indexs.stop_gradient = True
    vprobas = paddle.gather(probas, indexs[:, 0])
    vlabels = paddle.gather(labels, indexs[:, 0])
    return vprobas, vlabels

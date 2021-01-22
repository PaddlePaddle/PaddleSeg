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
    Multi-class Lovasz-Softmax loss

    Args:
        ignore_index: int64, specifies a target value that is ignored and does not contribute to the input gradient. Default ``None``.
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """

    def __init__(self, ignore_index=None, classes='present'):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index
        self.classes = classes

    def forward(self, logits, labels):
        """
        Forward computation.

        Args:
            logits: [N, C, H, W] Tensor, logits at each prediction (between -\infty and +\infty)
            labels: [N, 1, H, W] Tensor, ground truth labels (between 0 and C - 1)
        """
        probas = F.softmax(
            logits, axis=1
        )  # probas grad [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00  -5.5730345e-07]
        vprobas, vlabels = flatten_probas(
            probas, labels, self.ignore_index
        )  # vprobas grad和torch不同，第一个类别是0，torch不是，第一个类别的第一个元素是0.05
        loss = lovasz_softmax_flat(vprobas, vlabels, classes=self.classes)
        return loss


@manager.LOSSES.add_component
class LovaszHingeLoss(nn.Layer):
    """
    Binary Lovasz hinge loss

    Args:
        ignore_index: int64, specifies a target value that is ignored and does not contribute to the input gradient. Default ``None``.
    """

    def __init__(self, ignore_index=None):
        super(LovaszHingeLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        Forward computation.

        Args:
            logits: [N, C, H, W] Tensor, logits at each pixel (between -\infty and +\infty)
            labels: [N, 1, H, W] Tensor, binary ground truth masks (0 or 1)
        """
        loss = lovasz_hinge_flat(
            *flatten_binary_scores(logits, labels, self.ignore_index))
        return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = paddle.sum(gt_sorted)
    p = len(gt_sorted)

    intersection = gts - paddle.cumsum(gt_sorted, axis=0)
    union = gts + paddle.cumsum(1 - gt_sorted, axis=0)
    jaccard = 1.0 - intersection.cast('float32') / union.cast('float32')

    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        # jaccard0 = paddle.slice(jaccard, axis=[0], starts=[0], ends=[1])
        # jaccard1 = paddle.slice(jaccard, axis=[0], starts=[1], ends=[len_gt])
        # jaccard2 = paddle.slice(jaccard, axis=[0], starts=[0], ends=[-1])
        # jaccard = paddle.concat([jaccard0, jaccard1 - jaccard2], axis=0)
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Tensor, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels - 1.
    signs.stop_gradient = True
    errors = 1. - logits * signs
    errors_sorted, perm = paddle.fluid.core.ops.argsort(errors, 'axis', 0,
                                                        'descending', True)
    errors_sorted.stop_gradient = False
    gt_sorted = paddle.gather(labels, perm)
    grad = lovasz_grad(gt_sorted)
    grad.stop_gradient = True
    loss = paddle.dot(F.relu(errors_sorted), grad)
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels according to 'ignore'
    """
    scores = paddle.reshape(scores, [-1, 1])
    labels = paddle.reshape(labels, [-1, 1])
    labels.stop_gradient = True
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    # ignore = paddle.cast(ignore, 'int32')
    valid_mask = paddle.reshape(valid, (-1, 1))
    indexs = paddle.nonzero(valid_mask)
    indexs.stop_gradient = True
    vscores = paddle.gather(scores, indexs[:, 0])
    vlabels = paddle.gather(labels, indexs[:, 0])
    vscores = paddle.squeeze(vscores, axis=1)
    vlabels = paddle.squeeze(vlabels, axis=1)
    return vscores, vlabels


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Tensor, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
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
        errors = paddle.abs(fg - class_pred)  # errors梯度不同
        errors_sorted, perm = paddle.fluid.core.ops.argsort(
            errors, 'axis', 0, 'descending', True)
        errors_sorted.stop_gradient = False  # errors_sorted梯度相同

        fg_sorted = paddle.gather(fg, perm)
        fg_sorted.stop_gradient = True

        grad = lovasz_grad(fg_sorted)  # grad值相同，无梯度
        grad.stop_gradient = True
        loss = paddle.dot(errors_sorted, grad)
        losses.append(loss)  # loss梯度相同，值相同

    if len(classes_to_sum) == 1:
        return losses[0]

    losses_tensor = paddle.stack(losses)  # losses_tensor值相同，梯度也相同
    mean_loss = paddle.mean(losses_tensor)
    return mean_loss


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if len(probas.shape) == 3:
        probas = paddle.unsqueeze(probas, axis=1)
    C = probas.shape[1]
    probas = paddle.transpose(probas, [0, 2, 3, 1])
    probas = paddle.reshape(probas, [-1, C])
    labels = paddle.reshape(labels, [-1, 1])
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    # valid = paddle.cast(valid, 'int32')
    valid_mask = paddle.reshape(valid, [-1, 1])
    indexs = paddle.nonzero(valid_mask)
    indexs.stop_gradient = True
    vprobas = paddle.gather(probas, indexs[:, 0])
    # print(probas.shape, vprobas.shape)  # [1789832, 20] [1700971, 20]
    vlabels = paddle.gather(labels, indexs[:, 0])
    vlabels = paddle.squeeze(vlabels, axis=1)
    return vprobas, vlabels

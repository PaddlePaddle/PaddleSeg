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
import paddle.nn.functional as F


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = paddle.sum(gt_sorted)
    p = len(gt_sorted)

    intersection = gts - paddle.cumsum(gt_sorted, axis=0)
    union = gts + paddle.cumsum(1 - gt_sorted, axis=0)
    jaccard = 1.0 - intersection / union

    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        # jaccard0 = paddle.slice(jaccard, axis=[0], starts=[0], ends=[1])
        # jaccard1 = paddle.slice(jaccard, axis=[0], starts=[1], ends=[len_gt])
        # jaccard2 = paddle.slice(jaccard, axis=[0], starts=[0], ends=[-1])
        # jaccard = paddle.concat([jaccard0, jaccard1 - jaccard2], axis=0)
    return jaccard


def lovasz_hinge(logits, labels, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [N, C, H, W] Tensor, logits at each pixel (between -\infty and +\infty)
      labels: [N, 1, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: [N, 1, H, W] Tensor. Void class labels, ignore pixels which value=0
    """
    loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Tensor, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    signs.stop_gradient = True
    errors = 1. - logits * signs
    errors_sorted, perm = paddle.fluid.core.ops.argsort(errors, 'axis', 0,
                                                        'descending', True)
    perm = perm.data
    errors_sorted.stop_gradient = False
    gt_sorted = labels[perm]
    # gt_sorted = paddle.gather(labels, perm)
    grad = lovasz_grad(gt_sorted)
    grad.stop_gradient = True
    loss = paddle.dot(F.relu(errors_sorted), grad)
    return loss

    # shape = paddle.shape(logits)
    # y = paddle.zeros_like(shape[0])

    # out_var = paddle.create_tensor("float32")
    # with paddle.control_flow.Switch() as switch:
    #     with switch.case(paddle.equal(shape[0], y)):
    #         loss = paddle.sum(logits) * 0.
    #         paddle.assign(input=loss, output=out_var)
    #     with switch.case(paddle.greater_than(shape[0], y)):
    #         labelsf = paddle.cast(labels, logits.dtype)
    #         signs = labelsf * 2 - 1.
    #         signs.stop_gradient = True
    #         errors = 1.0 - paddle.elementwise_mul(logits, signs)
    #         errors_sorted, perm = paddle.argsort(
    #             errors, axis=0, descending=True)
    #         errors_sorted.stop_gradient = False
    #         gt_sorted = paddle.gather(labelsf, perm)

    #         grad = lovasz_grad(gt_sorted)
    #         grad.stop_gradient = True
    #         loss = paddle.sum(
    #             paddle.relu(errors_sorted) * grad)
    #         paddle.assign(input=loss, output=out_var)
    # return out_var


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
    ignore = paddle.cast(ignore, 'int32')
    ignore_mask = paddle.reshape(ignore, (-1, 1))
    indexs = paddle.where(ignore_mask == 1)
    indexs.stop_gradient = True
    vscores = paddle.gather(scores, indexs[:, 0])
    vlabels = paddle.gather(labels, indexs[:, 0])
    return vscores, vlabels


def lovasz_softmax(logits, labels, classes='present', ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      logits: [N, C, H, W] Tensor, logits at each prediction (between -\infty and +\infty)
      labels: [N, 1, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      ignore: [N, 1, H, W] Tensor. Void class labels, ignore pixels which value=0
    """
    probas = F.softmax(logits, axis=1)
    vprobas, vlabels = flatten_probas(probas, labels, ignore)
    loss = lovasz_softmax_flat(vprobas, vlabels, classes=classes)
    return loss


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
    present = []
    classes_to_sum = list(range(C)) if classes in ['all', 'present'
                                                   ] else classes
    for c in classes_to_sum:
        fg = paddle.cast(labels == c, probas.dtype)
        fg.stop_gradient = True
        if classes == 'present':
            present.append(paddle.cast(paddle.sum(fg) > 0, "int64"))
        if C == 1:
            if len(classes_to_sum) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = paddle.abs(fg - class_pred)
        errors_sorted, perm = paddle.fluid.core.ops.argsort(
            errors, 'axis', 0, 'descending', True)
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
    if classes == 'present':
        present_tensor = paddle.stack(present)
        index = paddle.nonzero(present_tensor)
        index.stop_gradient = True
        losses_tensor = paddle.gather(losses_tensor, index[:, 0])
    loss = paddle.mean(losses_tensor)
    return loss


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
    ignore = paddle.cast(ignore, 'int32')
    ignore_mask = paddle.reshape(ignore, [-1, 1])
    indexs = paddle.nonzero(ignore_mask)
    indexs.stop_gradient = True
    vprobas = paddle.gather(probas, indexs[:, 0])
    vlabels = paddle.gather(labels, indexs[:, 0])
    vlabels = paddle.squeeze(vlabels, axis=1)
    return vprobas, vlabels

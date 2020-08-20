# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.fluid as fluid
import numpy as np


def _cumsum(x):
    y = np.array(x)
    return np.cumsum(y, axis=0)


def create_tmp_var(name, dtype, shape):
    return fluid.default_main_program().current_block().create_var(
        name=name, dtype=dtype, shape=shape)


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gt_sorted = fluid.layers.squeeze(gt_sorted, axes=[1])
    gts = fluid.layers.reduce_sum(gt_sorted)
    len_gt = fluid.layers.shape(gt_sorted)

    # Acceleration is achieved by reducing the number of calls to cumsum.
    # This calculation method is equivalent to that of the original paper.
    var_one = fluid.layers.fill_constant(shape=[1], value=1, dtype='int32')
    range_ = fluid.layers.range(1, len_gt + var_one, 1, 'int32')
    tmp_var = create_tmp_var(
        name='tmp_var', dtype=gt_sorted.dtype, shape=gt_sorted.shape)
    cumsum_ = fluid.layers.py_func(func=_cumsum, x=gt_sorted, out=tmp_var)
    intersection = gts - cumsum_
    union = intersection + range_

    jaccard = 1.0 - intersection / union
    jaccard0 = fluid.layers.slice(jaccard, axes=[0], starts=[0], ends=[1])
    jaccard1 = fluid.layers.slice(jaccard, axes=[0], starts=[1], ends=[len_gt])
    jaccard2 = fluid.layers.slice(jaccard, axes=[0], starts=[0], ends=[-1])
    jaccard = fluid.layers.concat([jaccard0, jaccard1 - jaccard2], axis=0)
    jaccard = fluid.layers.unsqueeze(jaccard, axes=[1])
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
    shape = fluid.layers.shape(logits)
    y = fluid.layers.zeros_like(shape[0])

    out_var = fluid.layers.create_tensor("float32")
    with fluid.layers.control_flow.Switch() as switch:
        with switch.case(fluid.layers.equal(shape[0], y)):
            loss = fluid.layers.reduce_sum(logits) * 0.
            fluid.layers.assign(input=loss, output=out_var)
        with switch.case(fluid.layers.greater_than(shape[0], y)):
            labelsf = fluid.layers.cast(labels, logits.dtype)
            signs = labelsf * 2 - 1.
            signs.stop_gradient = True
            errors = 1.0 - fluid.layers.elementwise_mul(logits, signs)
            errors_sorted, perm = fluid.layers.argsort(
                errors, axis=0, descending=True)
            errors_sorted.stop_gradient = False
            gt_sorted = fluid.layers.gather(labelsf, perm)

            grad = lovasz_grad(gt_sorted)
            grad.stop_gradient = True
            loss = fluid.layers.reduce_sum(
                fluid.layers.relu(errors_sorted) * grad)
            fluid.layers.assign(input=loss, output=out_var)
    return out_var


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels according to 'ignore'
    """
    scores = fluid.layers.reshape(scores, [-1, 1])
    labels = fluid.layers.reshape(labels, [-1, 1])
    labels.stop_gradient = True
    if ignore is None:
        return scores, labels
    ignore = fluid.layers.cast(ignore, 'int32')
    ignore_mask = fluid.layers.reshape(ignore, (-1, 1))
    indexs = fluid.layers.where(ignore_mask == 1)
    indexs.stop_gradient = True
    vscores = fluid.layers.gather(scores, indexs[:, 0])
    vlabels = fluid.layers.gather(labels, indexs[:, 0])
    return vscores, vlabels


def lovasz_softmax(probas, labels, classes='present', ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [N, C, H, W] Tensor, class probabilities at each prediction (between 0 and 1).
      labels: [N, 1, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      ignore: [N, 1, H, W] Tensor. Void class labels, ignore pixels which value=0
    """
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
    C = probas.shape[1]
    losses = []
    present = []
    classes_to_sum = list(range(C)) if classes in ['all', 'present'
                                                   ] else classes
    for c in classes_to_sum:
        fg = fluid.layers.cast(labels == c, probas.dtype)
        fg.stop_gradient = True
        if classes == 'present':
            present.append(
                fluid.layers.cast(fluid.layers.reduce_sum(fg) > 0, "int64"))
        if C == 1:
            if len(classes_to_sum) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = fluid.layers.abs(fg - class_pred)
        errors_sorted, perm = fluid.layers.argsort(
            errors, axis=0, descending=True)
        errors_sorted.stop_gradient = False

        fg_sorted = fluid.layers.gather(fg, perm)
        fg_sorted.stop_gradient = True

        grad = lovasz_grad(fg_sorted)
        grad.stop_gradient = True
        loss = fluid.layers.reduce_sum(errors_sorted * grad)

        losses.append(loss)

    if len(classes_to_sum) == 1:
        return losses[0]

    losses_tensor = fluid.layers.stack(losses)
    if classes == 'present':
        present_tensor = fluid.layers.stack(present)
        index = fluid.layers.where(present_tensor == 1)
        index.stop_gradient = True
        losses_tensor = fluid.layers.gather(losses_tensor, index[:, 0])
    loss = fluid.layers.mean(losses_tensor)
    return loss


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if len(probas.shape) == 3:
        probas = fluid.layers.unsqueeze(probas, axis=[1])
    C = probas.shape[1]
    probas = fluid.layers.transpose(probas, [0, 2, 3, 1])
    probas = fluid.layers.reshape(probas, [-1, C])
    labels = fluid.layers.reshape(labels, [-1, 1])
    if ignore is None:
        return probas, labels
    ignore = fluid.layers.cast(ignore, 'int32')
    ignore_mask = fluid.layers.reshape(ignore, [-1, 1])
    indexs = fluid.layers.where(ignore_mask == 1)
    indexs.stop_gradient = True
    vprobas = fluid.layers.gather(probas, indexs[:, 0])
    vlabels = fluid.layers.gather(labels, indexs[:, 0])
    return vprobas, vlabels

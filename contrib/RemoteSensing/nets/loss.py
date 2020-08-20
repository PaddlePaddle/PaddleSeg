# coding: utf8
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

import paddle.fluid as fluid
import numpy as np


def softmax_with_loss(logit,
                      label,
                      ignore_mask=None,
                      num_classes=2,
                      weight=None,
                      ignore_index=255):
    ignore_mask = fluid.layers.cast(ignore_mask, 'float32')
    label = fluid.layers.elementwise_min(
        label, fluid.layers.assign(np.array([num_classes - 1], dtype=np.int32)))
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    logit = fluid.layers.reshape(logit, [-1, num_classes])
    label = fluid.layers.reshape(label, [-1, 1])
    label = fluid.layers.cast(label, 'int64')
    ignore_mask = fluid.layers.reshape(ignore_mask, [-1, 1])
    if weight is None:
        loss, probs = fluid.layers.softmax_with_cross_entropy(
            logit, label, ignore_index=ignore_index, return_softmax=True)
    else:
        label = fluid.layers.squeeze(label, axes=[-1])
        label_one_hot = fluid.one_hot(input=label, depth=num_classes)
        if isinstance(weight, list):
            assert len(
                weight
            ) == num_classes, "weight length must equal num of classes"
            weight = fluid.layers.assign(np.array([weight], dtype='float32'))
        elif isinstance(weight, str):
            assert weight.lower(
            ) == 'dynamic', 'if weight is string, must be dynamic!'
            tmp = []
            total_num = fluid.layers.cast(
                fluid.layers.shape(label)[0], 'float32')
            for i in range(num_classes):
                cls_pixel_num = fluid.layers.reduce_sum(label_one_hot[:, i])
                ratio = total_num / (cls_pixel_num + 1)
                tmp.append(ratio)
            weight = fluid.layers.concat(tmp)
            weight = weight / fluid.layers.reduce_sum(weight) * num_classes
        elif isinstance(weight, fluid.layers.Variable):
            pass
        else:
            raise ValueError(
                'Expect weight is a list, string or Variable, but receive {}'.
                format(type(weight)))
        weight = fluid.layers.reshape(weight, [1, num_classes])
        weighted_label_one_hot = fluid.layers.elementwise_mul(
            label_one_hot, weight)
        probs = fluid.layers.softmax(logit)
        loss = fluid.layers.cross_entropy(
            probs,
            weighted_label_one_hot,
            soft_label=True,
            ignore_index=ignore_index)
        weighted_label_one_hot.stop_gradient = True

    loss = loss * ignore_mask
    avg_loss = fluid.layers.mean(loss) / (
        fluid.layers.mean(ignore_mask) + 0.00001)

    label.stop_gradient = True
    ignore_mask.stop_gradient = True
    return avg_loss


# to change, how to appicate ignore index and ignore mask
def dice_loss(logit, label, ignore_mask=None, epsilon=0.00001):
    if logit.shape[1] != 1 or label.shape[1] != 1 or ignore_mask.shape[1] != 1:
        raise Exception(
            "dice loss is only applicable to one channel classfication")
    ignore_mask = fluid.layers.cast(ignore_mask, 'float32')
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    label = fluid.layers.transpose(label, [0, 2, 3, 1])
    label = fluid.layers.cast(label, 'int64')
    ignore_mask = fluid.layers.transpose(ignore_mask, [0, 2, 3, 1])
    logit = fluid.layers.sigmoid(logit)
    logit = logit * ignore_mask
    label = label * ignore_mask
    reduce_dim = list(range(1, len(logit.shape)))
    inse = fluid.layers.reduce_sum(logit * label, dim=reduce_dim)
    dice_denominator = fluid.layers.reduce_sum(
        logit, dim=reduce_dim) + fluid.layers.reduce_sum(
            label, dim=reduce_dim)
    dice_score = 1 - inse * 2 / (dice_denominator + epsilon)
    label.stop_gradient = True
    ignore_mask.stop_gradient = True
    return fluid.layers.reduce_mean(dice_score)


def bce_loss(logit, label, ignore_mask=None, ignore_index=255):
    if logit.shape[1] != 1 or label.shape[1] != 1 or ignore_mask.shape[1] != 1:
        raise Exception("bce loss is only applicable to binary classfication")
    label = fluid.layers.cast(label, 'float32')
    loss = fluid.layers.sigmoid_cross_entropy_with_logits(
        x=logit, label=label, ignore_index=ignore_index,
        normalize=True)  # or False
    loss = fluid.layers.reduce_sum(loss)
    label.stop_gradient = True
    ignore_mask.stop_gradient = True
    return loss

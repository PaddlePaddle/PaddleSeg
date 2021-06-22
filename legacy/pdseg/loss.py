# coding: utf8
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

import sys

import paddle
import paddle.nn.functional as F
import numpy as np
from utils.config import cfg


def softmax_with_loss(logit,
                      label,
                      ignore_mask=None,
                      num_classes=2,
                      weight=None):
    ignore_mask = paddle.cast(ignore_mask, 'float32')
    label = paddle.minimum(
        label, paddle.assign(np.array([num_classes - 1], dtype=np.int32)))
    logit = paddle.transpose(logit, [0, 2, 3, 1])
    logit = paddle.reshape(logit, [-1, num_classes])
    label = paddle.reshape(label, [-1, 1])
    label = paddle.cast(label, 'int64')
    ignore_mask = paddle.reshape(ignore_mask, [-1, 1])
    if weight is None:
        loss, probs = F.softmax_with_cross_entropy(
            logit,
            label,
            ignore_index=cfg.DATASET.IGNORE_INDEX,
            return_softmax=True)
    else:
        label = paddle.squeeze(label, axes=[-1])
        label_one_hot = F.one_hot(input=label, num_classes=num_classes)
        if isinstance(weight, list):
            assert len(
                weight
            ) == num_classes, "weight length must equal num of classes"
            weight = paddle.assign(np.array([weight], dtype='float32'))
        elif isinstance(weight, str):
            assert weight.lower(
            ) == 'dynamic', 'if weight is string, must be dynamic!'
            tmp = []
            total_num = paddle.cast(paddle.shape(label)[0], 'float32')
            for i in range(num_classes):
                cls_pixel_num = paddle.sum(label_one_hot[:, i])
                ratio = total_num / (cls_pixel_num + 1)
                tmp.append(ratio)
            weight = paddle.concat(tmp)
            weight = weight / paddle.sum(weight) * num_classes
        elif isinstance(weight, paddle.Tensor):
            pass
        else:
            raise ValueError(
                'Expect weight is a list, string or Variable, but receive {}'.
                format(type(weight)))
        weight = paddle.reshape(weight, [1, num_classes])
        weighted_label_one_hot = label_one_hot * weight
        probs = F.softmax(logit)
        loss = F.cross_entropy(
            probs,
            weighted_label_one_hot,
            soft_label=True,
            ignore_index=cfg.DATASET.IGNORE_INDEX)
        weighted_label_one_hot.stop_gradient = True

    loss = loss * ignore_mask
    avg_loss = paddle.mean(loss) / (
        paddle.mean(ignore_mask) + cfg.MODEL.DEFAULT_EPSILON)

    label.stop_gradient = True
    ignore_mask.stop_gradient = True
    return avg_loss


# to change, how to appicate ignore index and ignore mask
def dice_loss(logit, label, ignore_mask=None, epsilon=0.00001):
    if logit.shape[1] != 1 or label.shape[1] != 1 or ignore_mask.shape[1] != 1:
        raise Exception(
            "dice loss is only applicable to one channel classfication")
    ignore_mask = paddle.cast(ignore_mask, 'float32')
    logit = paddle.transpose(logit, [0, 2, 3, 1])
    label = paddle.transpose(label, [0, 2, 3, 1])
    label = paddle.cast(label, 'int64')
    ignore_mask = paddle.transpose(ignore_mask, [0, 2, 3, 1])
    logit = F.sigmoid(logit)
    logit = logit * ignore_mask
    label = label * ignore_mask
    reduce_dim = list(range(1, len(logit.shape)))
    inse = paddle.sum(logit * label, dim=reduce_dim)
    dice_denominator = paddle.sum(
        logit, dim=reduce_dim) + paddle.sum(
            label, dim=reduce_dim)
    dice_score = 1 - inse * 2 / (dice_denominator + epsilon)
    label.stop_gradient = True
    ignore_mask.stop_gradient = True
    return paddle.mean(dice_score)


def bce_loss(logit, label, ignore_mask=None):
    if logit.shape[1] != 1 or label.shape[1] != 1 or ignore_mask.shape[1] != 1:
        raise Exception("bce loss is only applicable to binary classfication")
    label = paddle.cast(label, 'float32')
    loss = paddle.sigmoid_cross_entropy_with_logits(
        x=logit,
        label=label,
        ignore_index=cfg.DATASET.IGNORE_INDEX,
        normalize=True)  # or False
    loss = paddle.sum(loss)
    label.stop_gradient = True
    ignore_mask.stop_gradient = True
    return loss


def multi_softmax_with_loss(logits,
                            label,
                            ignore_mask=None,
                            num_classes=2,
                            weight=None):
    if isinstance(logits, tuple):
        avg_loss = 0
        for i, logit in enumerate(logits):
            if label.shape[2] != logit.shape[2] or label.shape[
                    3] != logit.shape[3]:
                logit_label = F.interpolate(
                    label, logit.shape[2:], mode='nearest', align_corners=False)
            else:
                logit_label = label
            logit_mask = (logit_label.astype('int32') !=
                          cfg.DATASET.IGNORE_INDEX).astype('int32')
            loss = softmax_with_loss(
                logit, logit_label, logit_mask, num_classes, weight=weight)
            avg_loss += cfg.MODEL.MULTI_LOSS_WEIGHT[i] * loss
    else:
        avg_loss = softmax_with_loss(
            logits, label, ignore_mask, num_classes, weight=weight)
    return avg_loss


def multi_dice_loss(logits, label, ignore_mask=None):
    if isinstance(logits, tuple):
        avg_loss = 0
        for i, logit in enumerate(logits):
            if label.shape[2] != logit.shape[2] or label.shape[
                    3] != logit.shape[3]:
                logit_label = paddle.fluid.layers.resize_nearest(
                    label, logit.shape[2:])
            else:
                logit_label = label
            logit_mask = (logit_label.astype('int32') !=
                          cfg.DATASET.IGNORE_INDEX).astype('int32')
            loss = dice_loss(logit, logit_label, logit_mask)
            avg_loss += cfg.MODEL.MULTI_LOSS_WEIGHT[i] * loss
    else:
        avg_loss = dice_loss(logits, label, ignore_mask)
    return avg_loss


def multi_bce_loss(logits, label, ignore_mask=None):
    if isinstance(logits, tuple):
        avg_loss = 0
        for i, logit in enumerate(logits):
            if label.shape[2] != logit.shape[2] or label.shape[
                    3] != logit.shape[3]:
                logit_label = paddle.fluid.layers.resize_nearest(
                    label, logit.shape[2:])
            else:
                logit_label = label
            logit_mask = (logit_label.astype('int32') !=
                          cfg.DATASET.IGNORE_INDEX).astype('int32')
            loss = bce_loss(logit, logit_label, logit_mask)
            avg_loss += cfg.MODEL.MULTI_LOSS_WEIGHT[i] * loss
    else:
        avg_loss = bce_loss(logits, label, ignore_mask)
    return avg_loss

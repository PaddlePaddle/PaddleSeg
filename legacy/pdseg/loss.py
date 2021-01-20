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
import importlib
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
    loss, probs = F.softmax_with_cross_entropy(
        logit,
        label,
        ignore_index=cfg.DATASET.IGNORE_INDEX,
        return_softmax=True)

    loss = loss * ignore_mask
    avg_loss = paddle.mean(loss) / (
        paddle.mean(ignore_mask) + cfg.MODEL.DEFAULT_EPSILON)

    label.stop_gradient = True
    ignore_mask.stop_gradient = True
    return avg_loss


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
                    label, logit.shape[2:], mode='nearest', align_corners=True)
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

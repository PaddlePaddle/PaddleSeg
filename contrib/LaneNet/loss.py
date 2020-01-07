# coding: utf8
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
from utils.config import cfg

def softmax_with_loss(logit, label, ignore_mask=None, num_classes=2, weight=None):
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
            logit,
            label,
            ignore_index=cfg.DATASET.IGNORE_INDEX,
            return_softmax=True)
    else:
        label_one_hot = fluid.layers.one_hot(input=label, depth=num_classes)
        if isinstance(weight, list):
            assert len(weight) == num_classes, "weight length must equal num of classes"
            weight = fluid.layers.assign(np.array([weight], dtype='float32'))
        elif isinstance(weight, fluid.layers.Variable):
            pass
        else:
            tmp = []
            total_num = fluid.layers.cast(fluid.layers.shape(label)[0], 'float32')
            for i in range(num_classes):
                cls_pixel_num = fluid.layers.reduce_sum(label_one_hot[:, i])
                ratio = total_num / (cls_pixel_num + 1)
                tmp.append(ratio)
            weight = fluid.layers.concat(tmp)
            weight = weight / fluid.layers.reduce_sum(weight) * num_classes
        weight = fluid.layers.reshape(weight, [1, num_classes])
        weighted_label_one_hot = fluid.layers.elementwise_mul(label_one_hot, weight)
        probs = fluid.layers.softmax(logit)
        loss = fluid.layers.cross_entropy(
            probs,
            weighted_label_one_hot,
            soft_label=True,
            ignore_index=cfg.DATASET.IGNORE_INDEX)
        weighted_label_one_hot.stop_gradient = True

    loss = loss * ignore_mask
    avg_loss = fluid.layers.mean(loss) / fluid.layers.mean(ignore_mask)

    label.stop_gradient = True
    ignore_mask.stop_gradient = True
    return avg_loss


def multi_softmax_with_loss(logits, label, ignore_mask=None, num_classes=2, weight=None):
    if isinstance(logits, tuple):
        avg_loss = 0
        for i, logit in enumerate(logits):
            logit_label = fluid.layers.resize_nearest(label, logit.shape[2:])
            logit_mask = (logit_label.astype('int32') !=
                          cfg.DATASET.IGNORE_INDEX).astype('int32')
            loss = softmax_with_loss(logit, logit_label, logit_mask,
                                     num_classes)
            avg_loss += cfg.MODEL.MULTI_LOSS_WEIGHT[i] * loss
    else:
        avg_loss = softmax_with_loss(logits, label, ignore_mask, num_classes, weight)
    return avg_loss


def unsorted_segment_sum(data, segment_ids, unique_labels, feature_dims):
    zeros = fluid.layers.fill_constant_batch_size_like(unique_labels, shape=[1, feature_dims],
                                                       dtype='float32', value=0)
    segment_ids = fluid.layers.unsqueeze(segment_ids, axes=[1])
    segment_ids.stop_gradient = True
    segment_sum = fluid.layers.scatter_nd_add(zeros, segment_ids, data)
    zeros.stop_gradient = True

    return segment_sum


def norm(x, axis=-1):
    distance = fluid.layers.reduce_sum(fluid.layers.abs(x), dim=axis, keep_dim=True)
    return distance

def discriminative_loss_single(
        prediction,
        correct_label,
        feature_dim,
        label_shape,
        delta_v,
        delta_d,
        param_var,
        param_dist,
        param_reg):

    correct_label = fluid.layers.reshape(
        correct_label, [
            label_shape[1] * label_shape[0]])
    prediction = fluid.layers.transpose(prediction, [1, 2, 0])
    reshaped_pred = fluid.layers.reshape(
        prediction, [
            label_shape[1] * label_shape[0], feature_dim])

    unique_labels, unique_id, counts = fluid.layers.unique_with_counts(correct_label)
    correct_label.stop_gradient = True
    counts = fluid.layers.cast(counts, 'float32')
    num_instances = fluid.layers.shape(unique_labels)

    segmented_sum = unsorted_segment_sum(
        reshaped_pred, unique_id, unique_labels, feature_dims=feature_dim)

    counts_rsp = fluid.layers.reshape(counts, (-1, 1))
    mu = fluid.layers.elementwise_div(segmented_sum, counts_rsp)
    counts_rsp.stop_gradient = True
    mu_expand = fluid.layers.gather(mu, unique_id)
    tmp = fluid.layers.elementwise_sub(mu_expand, reshaped_pred)

    distance = norm(tmp)
    distance = distance - delta_v

    distance_pos = fluid.layers.greater_equal(distance, fluid.layers.zeros_like(distance))
    distance_pos = fluid.layers.cast(distance_pos, 'float32')
    distance = distance * distance_pos

    distance = fluid.layers.square(distance)

    l_var = unsorted_segment_sum(distance, unique_id, unique_labels, feature_dims=1)
    l_var = fluid.layers.elementwise_div(l_var, counts_rsp)
    l_var = fluid.layers.reduce_sum(l_var)
    l_var = l_var / fluid.layers.cast(num_instances * (num_instances - 1), 'float32')

    mu_interleaved_rep = fluid.layers.expand(mu, [num_instances, 1])
    mu_band_rep = fluid.layers.expand(mu, [1, num_instances])
    mu_band_rep = fluid.layers.reshape(mu_band_rep, (num_instances * num_instances, feature_dim))

    mu_diff = fluid.layers.elementwise_sub(mu_band_rep, mu_interleaved_rep)

    intermediate_tensor = fluid.layers.reduce_sum(fluid.layers.abs(mu_diff), dim=1)
    intermediate_tensor.stop_gradient = True
    zero_vector = fluid.layers.zeros([1], 'float32')
    bool_mask = fluid.layers.not_equal(intermediate_tensor, zero_vector)
    temp = fluid.layers.where(bool_mask)
    mu_diff_bool = fluid.layers.gather(mu_diff, temp)

    mu_norm = norm(mu_diff_bool)
    mu_norm = 2. * delta_d - mu_norm
    mu_norm_pos = fluid.layers.greater_equal(mu_norm, fluid.layers.zeros_like(mu_norm))
    mu_norm_pos = fluid.layers.cast(mu_norm_pos, 'float32')
    mu_norm = mu_norm * mu_norm_pos
    mu_norm_pos.stop_gradient = True

    mu_norm = fluid.layers.square(mu_norm)

    l_dist = fluid.layers.reduce_mean(mu_norm)

    l_reg = fluid.layers.reduce_mean(norm(mu, axis=1))

    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg
    loss = l_var + l_dist + l_reg
    return loss, l_var, l_dist, l_reg


def discriminative_loss(prediction, correct_label, feature_dim, image_shape,
                        delta_v, delta_d, param_var, param_dist, param_reg):
    batch_size = int(cfg.BATCH_SIZE_PER_DEV)
    output_ta_loss = 0.
    output_ta_var = 0.
    output_ta_dist = 0.
    output_ta_reg = 0.
    for i in range(batch_size):
        disc_loss_single, l_var_single, l_dist_single, l_reg_single = discriminative_loss_single(
            prediction[i], correct_label[i], feature_dim, image_shape, delta_v, delta_d, param_var, param_dist,
            param_reg)
        output_ta_loss += disc_loss_single
        output_ta_var += l_var_single
        output_ta_dist += l_dist_single
        output_ta_reg += l_reg_single

    disc_loss = output_ta_loss / batch_size
    l_var = output_ta_var / batch_size
    l_dist = output_ta_dist / batch_size
    l_reg = output_ta_reg / batch_size
    return disc_loss, l_var, l_dist, l_reg



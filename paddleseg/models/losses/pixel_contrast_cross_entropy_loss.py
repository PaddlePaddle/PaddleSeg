# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class PixelContrastCrossEntropyLoss(nn.Layer):
    """
    The PixelContrastCrossEntropyLoss implementation based on PaddlePaddle.

    The original article refers to
    Wenguan Wang, Tianfei Zhou, et al. "Exploring Cross-Image Pixel Contrast for Semantic Segmentation"
    (https://arxiv.org/abs/2101.11939).

    Args:
        temperature (float, optional): Controling the numerical similarity of features. Default: 0.1.
        base_temperature (float, optional): Controling the numerical range of contrast loss. Default: 0.07.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default 255.
        max_samples (int, optional): Max sampling anchors. Default: 1024.
        max_views (int): Sampled samplers of a class. Default: 100.
    """
    def __init__(self,
                 temperature=0.1,
                 base_temperature=0.07,
                 ignore_index=255,
                 max_samples=1024,
                 max_views=100):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.ignore_index = ignore_index
        self.max_samples = max_samples
        self.max_views = max_views

    def _hard_anchor_sampling(self, X, y_hat, y):
        """
        Args:
            X (Tensor): reshaped feats, shape = [N, H * W, feat_channels]
            y_hat (Tensor): reshaped label, shape = [N, H * W]
            y (Tensor): reshaped predict, shape = [N, H * W]
        """
        batch_size, feat_dim = paddle.shape(X)[0], paddle.shape(X)[-1]
        classes = []
        total_classes = 0
        for i in range(batch_size):
            current_y = y_hat[i]
            current_classes = paddle.unique(current_y)
            current_classes = [
                x for x in current_classes if x != self.ignore_index
            ]
            current_classes = [
                x for x in current_classes
                if (current_y == x).nonzero().shape[0] > self.max_views
            ]

            classes.append(current_classes)
            total_classes += len(current_classes)

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = []
        y_ = paddle.zeros([total_classes], dtype='float32')

        X_ptr = 0
        for i in range(batch_size):
            this_y_hat = y_hat[i]
            current_y = y[i]
            current_classes = classes[i]

            for cls_id in current_classes:
                hard_indices = paddle.logical_and(
                    (this_y_hat == cls_id), (current_y != cls_id)).nonzero()
                easy_indices = paddle.logical_and(
                    (this_y_hat == cls_id), (current_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                else:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep

                indices = None
                if num_hard > 0:
                    perm = paddle.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]].reshape(
                        (-1, hard_indices.shape[-1]))
                    indices = hard_indices
                if num_easy > 0:
                    perm = paddle.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]].reshape(
                        (-1, easy_indices.shape[-1]))
                    if indices is None:
                        indices = easy_indices
                    else:
                        indices = paddle.concat((indices, easy_indices), axis=0)
                if indices is None:
                    raise UserWarning('hard sampling indice error')

                X_.append(paddle.index_select(X[i, :, :], indices.squeeze(1)))
                y_[X_ptr] = float(cls_id)
                X_ptr += 1
        X_ = paddle.stack(X_, axis=0)
        return X_, y_

    def _contrastive(self, feats_, labels_):
        """
        Args:
            feats_ (Tensor): sampled pixel, shape = [total_classes, n_view, feat_dim], total_classes = batch_size * single image classes
            labels_ (Tensor): label, shape = [total_classes]
        """
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.reshape((-1, 1))
        mask = paddle.equal(labels_, paddle.transpose(labels_,
                                                      [1, 0])).astype('float32')

        contrast_count = n_view
        contrast_feature = paddle.concat(paddle.unbind(feats_, axis=1), axis=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = paddle.matmul(
            anchor_feature, paddle.transpose(contrast_feature,
                                             [1, 0])) / self.temperature
        logits_max = paddle.max(anchor_dot_contrast, axis=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max

        mask = paddle.tile(mask, [anchor_count, contrast_count])
        neg_mask = 1 - mask

        logits_mask = 1 - paddle.eye(mask.shape[0]).astype('float32')
        mask = mask * logits_mask

        neg_logits = paddle.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = paddle.exp(logits)

        log_prob = logits - paddle.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def contrast_criterion(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1)
        labels = F.interpolate(labels, feats.shape[2:], mode='nearest')
        labels = labels.squeeze(1)

        batch_size = feats.shape[0]
        labels = labels.reshape((batch_size, -1))
        predict = predict.reshape((batch_size, -1))
        feats = paddle.transpose(feats, [0, 2, 3, 1])
        feats = feats.reshape((feats.shape[0], -1, feats.shape[-1]))

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)
        return loss

    def forward(self, preds, label):
        assert "seg" in preds, "The input of PixelContrastCrossEntropyLoss should include 'seg' output, but not found."
        assert "embed" in preds, "The input of PixelContrastCrossEntropyLoss should include 'embed' output, but not found."

        seg = preds['seg']
        embedding = preds['embed']

        predict = paddle.argmax(seg, axis=1)
        loss = self.contrast_criterion(embedding, label, predict)
        return loss

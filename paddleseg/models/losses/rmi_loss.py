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
"""rmi loss in PaddlePaddle"""
import numpy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager

_euler_num = 2.718281828
_pi = 3.14159265
_ln_2_pi = 1.837877
_CLIP_MIN = 1e-6
_CLIP_MAX = 1.0
_POS_ALPHA = 5e-4
_IS_SUM = 1


@manager.LOSSES.add_component
class RMILoss(nn.Layer):
    """
    Implements the Region Mutual Information(RMI) Loss（https://arxiv.org/abs/1910.12037） for Semantic Segmentation.
    Unlike vanilla rmi loss which contains Cross Entropy Loss, we disband them and only
    left the RMI-related parts.
    The motivation is to allow for a more flexible combination of losses during training.
    For example, by employing mixed loss to merge RMI Loss with Boostrap Cross Entropy Loss,
    we can achieve the online mining of hard examples together with attention to region information.
    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self,
                 num_classes=19,
                 rmi_radius=3,
                 rmi_pool_way=0,
                 rmi_pool_size=3,
                 rmi_pool_stride=3,
                 loss_weight_lambda=0.5,
                 ignore_index=255):
        super(RMILoss, self).__init__()

        self.num_classes = num_classes
        assert rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_radius = rmi_radius
        assert rmi_pool_way in [0, 1, 2, 3]
        self.rmi_pool_way = rmi_pool_way
        assert rmi_pool_size == rmi_pool_stride
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        self.weight_lambda = loss_weight_lambda
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        self.ignore_index = ignore_index

    def forward(self, logits_4D, labels_4D, do_rmi=True):
        """
        Forward computation.
        Args:
            logits (Tensor): Shape is [N, C, H, W], logits at each prediction (between -\infty and +\infty).
            labels (Tensor): Shape is [N, H, W], ground truth labels (between 0 and C - 1).
        """
        logits_4D = paddle.cast(logits_4D, dtype='float32')
        labels_4D = paddle.cast(labels_4D, dtype='float32')

        loss = self.forward_sigmoid(logits_4D, labels_4D, do_rmi=do_rmi)
        return loss

    def forward_sigmoid(self, logits_4D, labels_4D, do_rmi=False):
        """
        Using the sigmiod operation both.
        Args:
                logits_4D   :   [N, C, H, W], dtype=float32
                labels_4D   :   [N, H, W], dtype=long
                do_rmi          :       bool
        """
        label_mask_3D = labels_4D != self.ignore_index
        valid_onehot_labels_4D = paddle.cast(
            F.one_hot(
                paddle.cast(
                    labels_4D, dtype='int64') * paddle.cast(
                        label_mask_3D, dtype='int64'),
                num_classes=self.num_classes),
            dtype='float32')
        # label_mask_flat = paddle.cast(
        #     paddle.reshape(label_mask_3D, [-1]), dtype='float32')

        valid_onehot_labels_4D = valid_onehot_labels_4D * paddle.unsqueeze(
            label_mask_3D, axis=3)
        valid_onehot_labels_4D.stop_gradient = True
        probs_4D = F.sigmoid(logits_4D) * paddle.unsqueeze(
            label_mask_3D, axis=1) + _CLIP_MIN

        valid_onehot_labels_4D = paddle.transpose(valid_onehot_labels_4D,
                                                  [0, 3, 1, 2])
        valid_onehot_labels_4D.stop_gradient = True
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        return rmi_loss

    def inverse(self, x):
        return paddle.inverse(x)

    def rmi_lower_bound(self, labels_4D, probs_4D):
        """
        calculate the lower bound of the region mutual information.
        Args:
                labels_4D   :   [N, C, H, W], dtype=float32
                probs_4D    :   [N, C, H, W], dtype=float32
        """
        assert labels_4D.shape == probs_4D.shape, print(
            'shapes', labels_4D.shape, probs_4D.shape)

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(
                    labels_4D,
                    kernel_size=p,
                    stride=s,
                    padding=self.kernel_padding)
                probs_4D = F.max_pool2d(
                    probs_4D,
                    kernel_size=p,
                    stride=s,
                    padding=self.kernel_padding)
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(
                    labels_4D,
                    kernel_size=p,
                    stride=s,
                    padding=self.kernel_padding)
                probs_4D = F.avg_pool2d(
                    probs_4D,
                    kernel_size=p,
                    stride=s,
                    padding=self.kernel_padding)
            elif self.rmi_pool_way == 2:
                shape = labels_4D.shape
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(
                    labels_4D, size=[new_h, new_w], mode='nearest')
                probs_4D = F.interpolate(
                    probs_4D,
                    size=[new_h, new_w],
                    mode='bilinear',
                    align_corners=True)
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")

        label_shape = labels_4D.shape
        n, c = label_shape[0], label_shape[1]

        la_vectors, pr_vectors = self.map_get_pairs(
            labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0)

        la_vectors = paddle.reshape(la_vectors, [n, c, self.half_d, -1])
        la_vectors = paddle.cast(la_vectors, dtype='float64')
        la_vectors.stop_gradient = True

        pr_vectors = paddle.reshape(pr_vectors, [n, c, self.half_d, -1])
        pr_vectors = paddle.cast(pr_vectors, dtype='float64')

        diag_matrix = paddle.unsqueeze(
            paddle.unsqueeze(
                paddle.eye(self.half_d), axis=0), axis=0)
        la_vectors = la_vectors - paddle.mean(la_vectors, axis=3, keepdim=True)

        la_cov = paddle.matmul(la_vectors,
                               paddle.transpose(la_vectors, [0, 1, 3, 2]))
        pr_vectors = pr_vectors - paddle.mean(pr_vectors, axis=3, keepdim=True)
        pr_cov = paddle.matmul(pr_vectors,
                               paddle.transpose(pr_vectors, [0, 1, 3, 2]))

        pr_cov_inv = self.inverse(pr_cov + paddle.cast(
            diag_matrix, dtype='float64') * _POS_ALPHA)

        la_pr_cov = paddle.matmul(la_vectors,
                                  paddle.transpose(pr_vectors, [0, 1, 3, 2]))

        appro_var = la_cov - paddle.matmul(
            paddle.matmul(la_pr_cov, pr_cov_inv),
            paddle.transpose(la_pr_cov, [0, 1, 3, 2]))

        rmi_now = 0.5 * self.log_det_by_cholesky(appro_var + paddle.cast(
            diag_matrix, dtype='float64') * _POS_ALPHA)

        rmi_per_class = paddle.cast(
            paddle.mean(
                paddle.reshape(rmi_now, [-1, self.num_classes]), axis=0),
            dtype='float32')
        rmi_per_class = paddle.divide(rmi_per_class,
                                      paddle.to_tensor(float(self.half_d)))

        rmi_loss = paddle.sum(rmi_per_class) if _IS_SUM else paddle.mean(
            rmi_per_class)

        return rmi_loss

    def log_det_by_cholesky(self, matrix):
        """
        Args:
            matrix: matrix must be a positive define matrix.
                    shape [N, C, D, D].
        """

        chol = paddle.cholesky(matrix)
        diag = paddle.diagonal(chol, offset=0, axis1=-2, axis2=-1)
        chol = paddle.log(diag + 1e-8)

        return 2.0 * paddle.sum(chol, axis=-1)

    def map_get_pairs(self, labels_4D, probs_4D, radius=3, is_combine=True):
        """
        Args:
            labels_4D   :   labels, shape [N, C, H, W]
            probs_4D    :   probabilities, shape [N, C, H, W]
            radius      :   the square radius
        Return:
            tensor with shape [N, C, radius * radius, H - (radius - 1), W - (radius - 1)]
        """

        label_shape = labels_4D.shape
        h, w = label_shape[2], label_shape[3]
        new_h, new_w = h - (radius - 1), w - (radius - 1)
        la_ns = []
        pr_ns = []
        for y in range(0, radius, 1):
            for x in range(0, radius, 1):
                la_now = labels_4D[:, :, y:y + new_h, x:x + new_w]
                pr_now = probs_4D[:, :, y:y + new_h, x:x + new_w]
                la_ns.append(la_now)
                pr_ns.append(pr_now)

        if is_combine:
            pair_ns = la_ns + pr_ns
            p_vectors = paddle.stack(pair_ns, axis=2)
            return p_vectors
        else:
            la_vectors = paddle.stack(la_ns, axis=2)
            pr_vectors = paddle.stack(pr_ns, axis=2)
            return la_vectors, pr_vectors

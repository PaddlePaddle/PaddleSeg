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
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class PointCrossEntropyLoss(nn.Layer):
    """
    Implements the point cross entropy loss function.

    The original article refers to
    Kirillov A, Wu Y, He K, et al. "PointRend: Image Segmentation As Rendering."
    (https://arxiv.org/abs/1912.08193).

    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for
            the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining. Default ``1.0``.
        data_format (str, optional): The tensor format to use, 'NCHW' or 'NHWC'. Default ``'NCHW'``.
    """

    def __init__(self,
                 weight=None,
                 ignore_index=255,
                 top_k_percent_pixels=1.0,
                 data_format='NCHW',
                 align_corners=False):
        super(PointCrossEntropyLoss, self).__init__()
        if weight is not None:
            weight = paddle.to_tensor(weight, dtype='float32')
        self.weight = weight
        self.ignore_index = ignore_index
        self.top_k_percent_pixels = top_k_percent_pixels
        self.EPS = 1e-8
        self.data_format = data_format
        self.align_corners = align_corners

    def forward(self, logits, label, semantic_weights=None):
        """
        Forward computation.

        Args:
            logits (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (logit,points). logit'shape: [N, C, point_num]. logit'shape:[N, point_num, 2], where C is number of classes.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            semantic_weights (Tensor, optional): Weights about loss for each pixels, shape is the same as label. Default: None.
        """
        # for loss
        logit, points = logits  # [N, C, point_num],[N, point_num, 2]
        label = label.unsqueeze(1)  # [N,1,H,W]
        label = point_sample(
            label.astype('float32'),
            points,
            mode='nearest',
            align_corners=self.align_corners)  # [N, 1, point_num]
        label = paddle.squeeze(label, axis=1).astype('int64')  # [N, xx]

        channel_axis = 1 if self.data_format == 'NCHW' else -1
        if self.weight is not None and logit.shape[channel_axis] != len(
                self.weight):
            raise ValueError(
                'The number of weights = {} must be the same as the number of classes = {}.'
                .format(len(self.weight), logit.shape[1]))

        logit = paddle.transpose(logit, [0, 2, 1])
        no_ignore_label = label
        #no_ignore_label[label==self.ignore_index] = 0
        loss = F.cross_entropy(
            logit,
            no_ignore_label,
            ignore_index=self.ignore_index,
            reduction='none')

        mask = label != self.ignore_index
        mask = paddle.cast(mask, 'float32')

        loss = loss * mask
        if semantic_weights is not None:
            loss = loss * semantic_weights

        if self.weight is not None:
            _one_hot = F.one_hot(label, logit.shape[-1])
            _one_hot_weight = _one_hot * self.weight
            loss = loss * _one_hot_weight.argmax(-1)
            coef = paddle.sum(_one_hot_weight, axis=-1)
            #coef = paddle.ones_like(label)
        else:
            coef = paddle.ones_like(label)

        label.stop_gradient = True
        mask.stop_gradient = True
        if self.top_k_percent_pixels == 1.0:
            avg_loss = paddle.mean(loss) / (paddle.mean(mask * coef) + self.EPS)
            return avg_loss

        loss = loss.reshape((-1, ))
        top_k_pixels = int(self.top_k_percent_pixels * loss.numel())
        loss, indices = paddle.topk(loss, top_k_pixels)
        coef = coef.reshape((-1, ))
        coef = paddle.gather(coef, indices)
        coef.stop_gradient = True

        return loss.mean() / (paddle.mean(coef) + self.EPS)


def point_sample(input, points, align_corners=False, **kwargs):
    """A wrapper around :func:`grid_sample` to support 3D point_coords tensors
    Unlike :func:`torch.nn.functional.grid_sample` it assumes point_coords to
    lie inside ``[0, 1] x [0, 1]`` square.
    Args:
        input (Tensor): Feature map, shape (N, C, H, W).
        points (Tensor): Image based absolute point coordinates (normalized),
            range [0, 1] x [0, 1], shape (N, P, 2) or (N, Hgrid, Wgrid, 2).
        align_corners (bool): Whether align_corners. Default: False
    Returns:
        Tensor: Features of `point` on `input`, shape (N, C, P) or
            (N, C, Hgrid, Wgrid).
    """

    def denormalize(grid):
        """Denormalize input grid from range [0, 1] to [-1, 1]
        Args:
            grid (Tensor): The grid to be denormalize, range [0, 1].
        Returns:
            Tensor: Denormalized grid, range [-1, 1].
        """

        return grid * 2.0 - 1.0

    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = paddle.unsqueeze(points, axis=2)  # [2, 2048, 1, 2]
    output = F.grid_sample(
        input, denormalize(points), align_corners=align_corners, **kwargs)
    if add_dim:
        output = paddle.squeeze(output, axis=3)
    return output

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from util import misc


class NormalizedFocalLossSigmoid(nn.Layer):#这个有问题
    def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=True,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label
        
        
        sample_weight = sample_weight.astype('float32')

        if not self._from_logits:
            pred = F.sigmoid(pred)
        alpha = paddle.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = paddle.where(sample_weight.astype('bool'), 1.0 - paddle.abs(label - pred), paddle.ones_like(pred))
        beta = (1 - pt) ** self._gamma
        sw_sum = paddle.sum(sample_weight, axis=(-2, -1), keepdim=True)
        beta_sum = paddle.sum(beta, axis=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)

        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        with paddle.no_grad():
            ignore_area = paddle.sum((label == self._ignore_label).astype('float32'),
                                     axis=tuple(range(1, len(label.shape)))).numpy()
            sample_mult = paddle.mean(mult, axis=tuple(range(1, len(mult.shape)))).numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()
                beta_pmax = paddle.max(paddle.flatten(beta, 1), axis=1)
                beta_pmax = float(paddle.mean(beta_pmax))
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        loss_mask = pt + self._eps < 1
        loss_mask = loss_mask.astype('float32')
        pt_mask = (pt + self._eps) * loss_mask + (1 - loss_mask)* paddle.ones(pt.shape)
        loss = -alpha * beta * paddle.log(pt_mask)
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = paddle.sum(sample_weight,
                              axis=misc.get_dims_with_exclusion(len(sample_weight.shape), self._batch_axis))
            loss = paddle.sum(loss, axis=misc.get_dims_with_exclusion(len(loss.shape), self._batch_axis)) / (
                    bsum + self._eps)
        else:
            loss = paddle.sum(loss, axis=paddle.get_dims_with_exclusion(len(loss.shape), self._batch_axis))

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
        sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)


class FocalLoss(nn.Layer):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0,
                 ignore_label=-1):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = F.sigmoid(pred)
        alpha = paddle.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = paddle.where(one_hot, 1.0 - paddle.abs(label - pred), paddle.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * paddle.log(paddle.min(pt + self._eps, paddle.ones(1, dtype='float32')))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            tsum = paddle.sum(label == 1, axis=misc.get_dims_with_exclusion(len(label.shape), self._batch_axis))
            loss = paddle.sum(loss, axis=misc.get_dims_with_exclusion(len(loss.shape), self._batch_axis)) / (
                    tsum + self._eps)
        else:
            loss = paddle.sum(loss, axis=misc.get_dims_with_exclusion(len(loss.shape), self._batch_axis))
        return self._scale * loss


class SoftIoU(nn.Layer):
    def __init__(self, from_sigmoid=False, ignore_label=-1):
        super().__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def forward(self, pred, label):
        label = label.reshape(pred.shape)
        sample_weight = label != self._ignore_label

        if not self._from_sigmoid:
            pred = F.sigmoid(pred)

        loss = 1.0 - paddle.sum(pred * label * sample_weight, axis=(1, 2, 3)) \
               / (paddle.sum(paddle.max(pred, label) * sample_weight, axis=(1, 2, 3)) + 1e-8)

        return loss


class SigmoidBinaryCrossEntropyLoss(nn.Layer):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):

        label = label.reshape(pred.shape)
        sample_weight = label != self._ignore_label
        label = paddle.where(sample_weight, label, paddle.zeros_like(label))

        if not self._from_sigmoid:
            loss = F.relu(pred) - pred * label + F.softplus(-paddle.abs(pred))
        else:
            eps = 1e-12
            loss = -(paddle.log(pred + eps) * label + paddle.log(1. - pred + eps) * (1. - label))
        loss = self._weight * (loss * sample_weight)
        return paddle.mean(loss, axis=misc.get_dims_with_exclusion(len(loss.shape), self._batch_axis))

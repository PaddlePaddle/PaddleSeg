import paddle
import paddle.nn.functional as F
import numpy as np

from util import misc


class TrainMetric(object):
    def __init__(self, pred_outputs, gt_outputs):
        self.pred_outputs = pred_outputs
        self.gt_outputs = gt_outputs

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def get_epoch_value(self):
        raise NotImplementedError

    def reset_epoch_stats(self):
        raise NotImplementedError

    def log_states(self, sw, tag_prefix, global_step):
        pass

    @property
    def name(self):
        return type(self).__name__


class AdaptiveIoU(TrainMetric):
    def __init__(self, init_thresh=0.4, thresh_step=0.025, thresh_beta=0.99, iou_beta=0.9,
                 ignore_label=-1, from_logits=True,
                 pred_output='instances', gt_output='instances'):
        super().__init__(pred_outputs=(pred_output,), gt_outputs=(gt_output,))
        self._ignore_label = ignore_label
        self._from_logits = from_logits
        self._iou_thresh = init_thresh
        self._thresh_step = thresh_step
        self._thresh_beta = thresh_beta
        self._iou_beta = iou_beta
        self._ema_iou = 0.0
        self._epoch_iou_sum = 0.0
        self._epoch_batch_count = 0

    def update(self, pred, gt):
        gt_mask = gt > 0.5
        if self._from_logits:
            pred = F.sigmoid(pred)

        gt_mask_area = paddle.sum(gt_mask.astype('float32'), axis=(1, 2)).detach().numpy()
        if np.all(gt_mask_area == 0):
            return

        ignore_mask = gt == self._ignore_label
        max_iou = _compute_iou(pred > self._iou_thresh, gt_mask, ignore_mask).mean()
        best_thresh = self._iou_thresh
        for t in [best_thresh - self._thresh_step, best_thresh + self._thresh_step]:
            temp_iou = _compute_iou(pred > t, gt_mask, ignore_mask).mean()
            if temp_iou > max_iou:
                max_iou = temp_iou
                best_thresh = t

        self._iou_thresh = self._thresh_beta * self._iou_thresh + (1 - self._thresh_beta) * best_thresh
        self._ema_iou = self._iou_beta * self._ema_iou + (1 - self._iou_beta) * max_iou
        self._epoch_iou_sum += max_iou
        self._epoch_batch_count += 1

    def get_epoch_value(self):
        if self._epoch_batch_count > 0:
            return self._epoch_iou_sum / self._epoch_batch_count
        else:
            return 0.0

    def reset_epoch_stats(self):
        self._epoch_iou_sum = 0.0
        self._epoch_batch_count = 0

    def log_states(self, sw, tag_prefix, global_step):
        sw.add_scalar(tag_prefix + '_ema_iou', self._ema_iou, global_step)
        sw.add_scalar(tag_prefix + '_iou_thresh', self._iou_thresh, global_step)

    @property
    def iou_thresh(self):
        return self._iou_thresh


def _compute_iou(pred_mask, gt_mask, ignore_mask=None, keep_ignore=False):
    if ignore_mask is not None:

        pred_mask = paddle.where(ignore_mask, paddle.zeros_like(pred_mask.astype('float32')), pred_mask.astype('float32'))

    reduction_dims = misc.get_dims_with_exclusion(len(gt_mask.shape), 0)
    pred_mask = pred_mask.astype('bool')
    m = pred_mask.numpy() | gt_mask.numpy()
    n = pred_mask.numpy() & gt_mask.numpy()
    union = np.mean(m.astype(np.float), axis=tuple(reduction_dims))
    intersection = np.mean(n.astype(np.float), axis=tuple(reduction_dims))

    nonzero = union > 0

    iou = intersection[nonzero] / union[nonzero]
    if not keep_ignore:
        return iou
    else:
        result = np.full_like(intersection, -1)
        result[nonzero] = iou
        return result


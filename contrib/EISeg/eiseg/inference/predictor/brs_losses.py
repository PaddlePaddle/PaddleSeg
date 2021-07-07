import paddle
import paddle.nn as nn

from model.losses import SigmoidBinaryCrossEntropyLoss


class BRSMaskLoss(nn.Layer):
    def __init__(self, eps=1e-5):
        super().__init__()
        self._eps = eps

    def forward(self, result, pos_mask, neg_mask):
        pos_diff = (1 - result) * pos_mask
        pos_target = paddle.sum(pos_diff ** 2)
        pos_target = pos_target / (paddle.sum(pos_mask) + self._eps)

        neg_diff = result * neg_mask
        neg_target = paddle.sum(neg_diff ** 2)
        neg_target = neg_target / (paddle.sum(neg_mask) + self._eps)

        loss = pos_target + neg_target

        with paddle.no_grad():
            f_max_pos = paddle.max(paddle.abs(pos_diff))
            f_max_neg = paddle.max(paddle.abs(neg_diff))

        return loss, f_max_pos, f_max_neg


class OracleMaskLoss(nn.Layer):
    def __init__(self):
        super().__init__()
        self.gt_mask = None
        self.loss = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
        self.predictor = None
        self.history = []

    def set_gt_mask(self, gt_mask):
        self.gt_mask = gt_mask
        self.history = []

    def forward(self, result, pos_mask, neg_mask):
        gt_mask = self.gt_mask
        if self.predictor.object_roi is not None:
            r1, r2, c1, c2 = self.predictor.object_roi[:4]
            gt_mask = gt_mask[:, :, r1:r2 + 1, c1:c2 + 1]
            gt_mask = paddle.nn.functional.interpolate(gt_mask, result.size()[2:], mode='bilinear', align_corners=True)

        if result.shape[0] == 2:
            gt_mask_flipped = paddle.flip(gt_mask, axis=[3])
            gt_mask = paddle.concat([gt_mask, gt_mask_flipped], axis=0)

        loss = self.loss(result, gt_mask)
        self.history.append(loss.detach().cpu().numpy()[0])

        if len(self.history) > 5 and abs(self.history[-5] - self.history[-1]) < 1e-5:
            return 0, 0, 0

        return loss, 1.0, 1.0

import paddle
from paddle import nn
from paddleseg.utils import ramps
import paddle.nn.functional as F
from paddleseg.cvlibs import manager


# 无监督损失的权重
class ConsistencyWeight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """

    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends * iters_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        # 获得ramps的ramp_type的属性值，相当与ramps.ramp_type
        self.rampup_func = getattr(ramps, ramp_type)
        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
        return self.final_w * self.current_rampup


def CE_loss(input_logits, target_targets, ignore_index, temperature=1):
    if len(target_targets.shape) > 3:
        target_targets = paddle.argmax(target_targets, axis=1)
    return F.cross_entropy(input_logits / temperature, target_targets, ignore_index=ignore_index)


@manager.LOSSES.add_component
class SemiCeLoss(nn.Layer):
    def __init__(self, conf_mask=True, threshold=None, ignore_index=255,
                 threshold_neg=.0, temperature_value=1):
        super(SemiCeLoss, self).__init__()
        self.conf_mask = conf_mask
        self.threshold = threshold
        self.threshold_neg = threshold_neg
        self.temperature_value = temperature_value
        self.ignore_index = ignore_index

    def forward(self, logit, label):
        pass_rate = {}
        if self.conf_mask:
            # for negative
            targets_prob = F.softmax(label / self.temperature_value, axis=1)

            # for positive
            targets_real_prob = F.softmax(label, axis=1)

            weight = targets_real_prob.max(1)[0]
            total_number = len(targets_prob.flatten(0))
            boundary = ["< 0.1", "0.1~0.2", "0.2~0.3",
                        "0.3~0.4", "0.4~0.5", "0.5~0.6",
                        "0.6~0.7", "0.7~0.8", "0.8~0.9",
                        "> 0.9"]

            rate = [
                paddle.sum((paddle.logical_and((i - 1) / 10 < targets_real_prob, targets_real_prob < i / 10)) == True)
                / total_number for i in range(1, 11)]

            max_rate = [paddle.sum((paddle.logical_and((i - 1) / 10 < weight, weight < i / 10)) == True)
                        / weight.numel() for i in range(1, 11)]

            pass_rate["entire_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, rate)]
            pass_rate["max_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, max_rate)]

            mask = (weight >= self.threshold)

            mask_neg = (targets_prob < self.threshold_neg)

            neg_label = F.one_hot(paddle.argmax(targets_prob, axis=1), num_classes=logit.shape[1])
            neg_label = paddle.cast(neg_label, dtype=label.dtype)
            if neg_label.shape[-1] != 21:
                neg_label = paddle.concat([neg_label, paddle.zeros([neg_label.shape[0], neg_label.shape[1],
                                                                    neg_label.shape[2],
                                                                    21 - neg_label.shape[-1]]).cuda()],
                                          axis=3)
            neg_label = paddle.transpose(neg_label, perm=(0, 3, 1, 2))
            neg_label = 1 - neg_label

            if not paddle.any(mask):
                neg_prediction_prob = paddle.clip(1 - F.softmax(logit, axis=1), min=1e-7, max=1.)
                negative_loss_mat = -(neg_label * paddle.log(neg_prediction_prob))
                zero = paddle.to_tensor(0., dtype=paddle.float32, place=negative_loss_mat.place)
                return zero, pass_rate, negative_loss_mat[mask_neg].mean()
            else:
                positive_loss_mat = F.cross_entropy(logit, paddle.argmax(label, axis=1), reduction="none", axis=1)
                positive_loss_mat = positive_loss_mat * weight

                neg_prediction_prob = paddle.clip(1 - F.softmax(logit, axis=1), min=1e-7, max=1.)
                negative_loss_mat = -(neg_label * paddle.log(neg_prediction_prob))
                mask = paddle.expand(mask, shape=[positive_loss_mat.shape[0], positive_loss_mat.shape[1],
                                                  positive_loss_mat.shape[2]])
                return positive_loss_mat[mask].mean(), pass_rate, negative_loss_mat[mask_neg].mean()
        else:
            raise NotImplementedError

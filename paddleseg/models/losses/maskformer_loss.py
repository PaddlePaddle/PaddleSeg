# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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
#
# The implementation has referred to :https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/criterion.py

import copy
import numpy as np
from scipy.optimize import linear_sum_assignment

import paddle
import paddle.nn as nn
import paddle.distributed as dist
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = F.sigmoid(inputs)
    inputs = paddle.flatten(inputs, 1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha=0.25, gamma=2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = F.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_masks


def batch_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = F.sigmoid(inputs)
    inputs = paddle.flatten(inputs, start_axis=1)
    numerator = 2 * paddle.einsum("nc,mc->nm", inputs, targets)
    denominator = paddle.sum(inputs, axis=-1, keepdim=True) + paddle.sum(
        targets, axis=-1).unsqueeze(0)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    prob = F.sigmoid(inputs)
    focal_pos = ((1 - prob)**gamma) * F.binary_cross_entropy_with_logits(
        inputs, paddle.ones_like(inputs), reduction="none")
    focal_neg = (prob**gamma) * F.binary_cross_entropy_with_logits(
        inputs, paddle.zeros_like(inputs), reduction="none")
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = paddle.einsum("nc,mc->nm", focal_pos, targets) + paddle.einsum(
        "nc,mc->nm", focal_neg, (1 - targets))

    return loss / hw


class HungarianMatcher(nn.Layer):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class=1, cost_mask=1, cost_dice=1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @paddle.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching More memory-friendly.

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = F.softmax(
                outputs["pred_logits"][b],
                axis=-1)  # [num_queries, num_classes]
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]

            tgt_ids = targets[b]["labels"]
            # gt masks are already padded when preparing target
            if targets[b]["labels"].shape[0] == 0:
                indices.append((np.array(
                    [], dtype='int64'), np.array(
                        [], dtype='int64')))
                continue
            tgt_mask = paddle.cast(targets[b]["masks"], out_mask.dtype)
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.            
            cost_class = -paddle.gather(out_prob, index=tgt_ids, axis=1)

            # Downsample gt masks to save memory
            tgt_mask = F.interpolate(
                tgt_mask[:, None], size=out_mask.shape[-2:], mode="nearest")

            # Flatten spatial dimension
            out_mask = out_mask.flatten(1)  # [batch_size * num_queries, H*W]
            tgt_mask = tgt_mask[:, 0].flatten(1)  # [num_total_targets, H*W]

            # Compute the focal loss between masks
            cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)

            # Compute the dice loss betwen masks
            cost_dice = batch_dice_loss(out_mask, tgt_mask)

            # Final cost matrix
            C = (self.cost_mask * cost_mask + self.cost_class * cost_class +
                 self.cost_dice * cost_dice)
            C = C.reshape([num_queries, -1])

            indices.append(linear_sum_assignment(C))

        return [(paddle.to_tensor(
            i, dtype='int64'), paddle.to_tensor(
                j, dtype='int64')) for i, j in indices]


def nested_tensor_from_tensor_list(tensor_list):
    def _max_by_axis(the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        tensor = paddle.zeros(batch_shape, dtype=tensor_list[0].dtype)
        mask = paddle.ones((b, h, w), dtype="bool")

        for i in range(tensor.shape[0]):
            img = tensor_list[i]
            tensor[i, :img.shape[0], :img.shape[1], :img.shape[
                2]] = copy.deepcopy(img)
            mask[i, :img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return tensor, mask


@manager.LOSSES.add_component
class MaskFormerLoss(nn.Layer):
    """
    The Maskformer loss implemeted with PaddlePaddle.

    Args:
        num_classes(int): The number of classes that you want this network to classify. Default:150.
        eos_coef(float): The weight coefficient of the last class. Default: 0.1.
        losses(Tuple): The category of losses that you want to compute. Default: ("labels", 'masks').
        ignore_index(int): The ignored label when we calculate the loss. Default:255.

    """

    def __init__(self,
                 num_classes=150,
                 eos_coef=0.1,
                 losses=("labels", 'masks'),
                 ignore_index=255):
        super().__init__()
        mask_weight = 20.0
        dice_weight = 1.0

        weight_dict = {
            "loss_ce": 1,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight
        }
        eos_coef = 0.1
        dec_layers = 6
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update(
                {k + f"_{i}": v
                 for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.weight_dict = weight_dict
        self.matcher = HungarianMatcher(
            cost_class=1, cost_mask=mask_weight, cost_dice=dice_weight)
        self.losses = losses
        self.empty_weight = paddle.ones(shape=(num_classes + 1, ))
        self.empty_weight[-1] = eos_coef

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        targets_cpt, indices_cpt = [], []
        for t, indice in zip(targets, indices):
            if t['labels'].shape[0] != 0:
                targets_cpt.append(t)
                indices_cpt.append(indice)
        else:
            if indices_cpt == []:
                losses = {"loss_ce": paddle.to_tensor([0.0])}
                return losses

        assert "pred_logits" in outputs, "The 'pred_logits' need to be in outputs, but only got keys: {}".format(
            outputs.keys())

        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = paddle.concat(
            [t["labels"][J] for t, (_, J) in zip(targets_cpt, indices_cpt)])
        target_classes = paddle.full(
            src_logits.shape[:2], self.num_classes, dtype='int64')
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose((0, 2, 1)).cast('float32'),
            target_classes,
            weight=self.empty_weight,
            axis=1,
            use_softmax=True,
            ignore_index=255)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs, "The 'pred_masks' need to be in outputs, but only got keys: {}".format(
            outputs.keys())

        targets_cpt, indices_cpt = [], []
        for t, indice in zip(targets, indices):
            if t['labels'].shape[0] != 0:
                targets_cpt.append(t)
                indices_cpt.append(indice)
        else:
            if indices_cpt == []:
                losses = {
                    "loss_mask": paddle.to_tensor([0.0]),
                    "loss_dice": paddle.to_tensor([0.0]),
                }
                return losses
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices_cpt)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        if src_masks.ndim == 2:
            src_masks = src_masks.unsqueeze(0)
        masks = [t["masks"] for t in targets_cpt]

        target_masks, valid = nested_tensor_from_tensor_list(masks)
        target_masks = paddle.cast(target_masks, src_masks.dtype)
        target_masks = target_masks[tgt_idx]

        src_masks = F.interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False)
        src_masks = paddle.flatten(src_masks[:, 0], 1)

        target_masks = paddle.flatten(target_masks, 1)
        target_masks = target_masks.reshape(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(src_masks, target_masks, num_masks),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = paddle.concat(
            [paddle.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = paddle.concat([src for (src, _) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = paddle.concat(
            [paddle.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = paddle.concat([tgt for (_, tgt) in indices])

        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, logits, targets):
        targets_cpt = []
        batch_size = targets['gt_masks'].shape[0]
        # split targets in a batch
        for target_per_image_idx in range(batch_size):
            gt_masks = targets['gt_masks'][target_per_image_idx, ...]
            padded_masks = paddle.zeros(
                (gt_masks.shape[0], gt_masks.shape[-2], gt_masks.shape[-1]),
                dtype=gt_masks.dtype)
            padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks

            targets_cpt.append({
                "labels": targets['gt_classes'][target_per_image_idx, ...],
                "masks": padded_masks
            })

        targets = []
        for item in targets_cpt:
            item['masks'] = paddle.cast(item['masks'], 'bool')
            invalid_indices = paddle.nonzero(
                paddle.cast(item['labels'] == self.ignore_index, 'int64'))
            if len(invalid_indices) > 0:
                start_idx = int(invalid_indices[0].numpy())
            else:
                start_idx = len(item['labels'])
            index = paddle.cast(
                paddle.to_tensor([i for i in range(start_idx)]), 'int64')
            item['labels'] = paddle.gather(
                item['labels'], index, axis=0)  # [n] n<150
            item['masks'] = paddle.gather(
                item["masks"], index, axis=0)  # [n,512,512]
            targets.append(item)

        logits_without_aux = {
            k: v
            for k, v in logits.items() if k != "aux_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(logits_without_aux, targets)

        num_masks = sum(len(t['labels']) for t in targets)
        num_masks = paddle.to_tensor([num_masks], dtype='float32')

        if dist.get_world_size() > 1:
            dist.all_reduce(num_masks)
        num_masks = paddle.clip(
            num_masks / dist.get_world_size(), min=1).detach().numpy()[0]

        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, logits_without_aux, targets, indices,
                              num_masks))

        if "aux_outputs" in logits:
            for i in range(len(logits['aux_outputs'])):
                indices = self.matcher(logits['aux_outputs'][i], targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, logits['aux_outputs'][i],
                                           targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        for k in list(losses.keys()):
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]
            else:
                losses.pop(k)

        return sum(losses.values())

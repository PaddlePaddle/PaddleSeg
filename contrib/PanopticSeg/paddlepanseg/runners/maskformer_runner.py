# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from scipy.optimize import linear_sum_assignment

from paddlepanseg.cvlibs import manager, build_info_dict
from paddlepanseg.core.runner import PanSegRunner


def point_sample(input, point_coords, **grid_sample_kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0,
                           **grid_sample_kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def d2_get_uncertain_point_coords_with_randomness(
        coarse_logits, uncertainty_func, num_points, oversample_ratio,
        importance_sample_ratio):
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = paddle.rand((num_boxes, num_sampled, 2))
    point_logits = point_sample(
        coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = paddle.topk(
        point_uncertainties[:, 0, :], k=num_uncertain_points, axis=1)[1]
    shift = num_sampled * paddle.arange(num_boxes, dtype='int64')
    idx += shift[:, None]
    point_coords = point_coords.reshape((-1, 2))
    point_coords = paddle.index_select(point_coords, idx.flatten(), axis=0)
    point_coords = point_coords.reshape((num_boxes, num_uncertain_points, 2))
    if num_random_points > 0:
        point_coords = paddle.concat(
            [
                point_coords,
                paddle.rand((num_boxes, num_random_points, 2)),
            ],
            axis=1, )
    return point_coords


def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
    """
    inputs = F.sigmoid(inputs)
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def batch_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
    """
    inputs = F.sigmoid(inputs)
    inputs = inputs.flatten(1)
    numerator = 2 * paddle.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def sigmoid_ce_loss(inputs, targets, num_masks):
    """
    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
    
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    return loss.mean(1).sum() / num_masks


def batch_sigmoid_ce_loss(inputs, targets):
    """
    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, paddle.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(
        inputs, paddle.zeros_like(inputs), reduction='none')

    loss = paddle.einsum('nc,mc->nm', pos, targets) + paddle.einsum(
        'nc,mc->nm', neg, (1 - targets))

    return loss / hw


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(paddle.abs(gt_class_logits))


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        tensor = paddle.zeros(batch_shape, dtype=dtype)
        mask = paddle.ones((b, h, w), dtype='bool')
        # Automatically pad images and masks
        for i in range(b):
            img = tensor_list[i]
            tensor[i, :img.shape[0], :img.shape[1], :img.shape[2]] = img
            mask[i, :img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


class HungarianMatcher(nn.Layer):
    """
    This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class, cost_mask, cost_dice, num_points):
        """
        Creates the matcher

        Args:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @paddle.no_grad()
    def memory_efficient_forward(self, sample_dict, net_out_dict):
        """More memory-friendly matching"""
        bs, num_queries = net_out_dict['logits'].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = F.softmax(
                net_out_dict['logits'][b],
                axis=-1)  # [num_queries, num_classes]
            tgt_ids = sample_dict['gt_ids'][b]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -paddle.index_select(out_prob, tgt_ids, axis=1)

            out_mask = net_out_dict['masks'][b]  # [num_queries, H_pred, W_pred]
            # GT masks are already padded when preparing target
            tgt_mask = sample_dict['gt_masks'][b]

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # All masks share the same set of points for efficient matching!
            point_coords = paddle.rand((1, self.num_points, 2))
            # Get GT labels
            tgt_mask = point_sample(
                tgt_mask,
                paddle.tile(point_coords, (tgt_mask.shape[0], 1, 1)),
                align_corners=False, ).squeeze(1)
            out_mask = point_sample(
                out_mask,
                paddle.tile(point_coords, (out_mask.shape[0], 1, 1)),
                align_corners=False, ).squeeze(1)

            # Disable mixed-precision
            with paddle.amp.auto_cast(enable=False):
                out_mask = out_mask.astype('float32')
                tgt_mask = tgt_mask.astype('float32')

                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss(out_mask, tgt_mask)

            # Final cost matrix
            C = (self.cost_mask * cost_mask + self.cost_class * cost_class +
                 self.cost_dice * cost_dice)
            # Reshape and convert to numpy array
            C = C.reshape((num_queries, -1)).numpy()

            indices.append(linear_sum_assignment(C))

        return [(paddle.to_tensor(
            i, dtype='int64'), paddle.to_tensor(
                j, dtype='int64')) for i, j in indices]

    @paddle.no_grad()
    def forward(self, sample_dict, net_out_dict):
        return self.memory_efficient_forward(sample_dict, net_out_dict)


@manager.RUNNERS.add_component
class MaskFormerRunner(PanSegRunner):
    def __init__(self, num_classes, weight_ce, weight_mask, weight_dice,
                 eos_coef, num_points, oversample_ratio,
                 importance_sample_ratio):
        super().__init__()
        self.num_classes = num_classes
        self.weight_ce = weight_ce
        self.weight_mask = weight_mask
        self.weight_dice = weight_dice

        # Use a Hungarian matcher
        self.matcher = HungarianMatcher(
            cost_class=weight_ce,
            cost_mask=weight_mask,
            cost_dice=weight_dice,
            num_points=num_points)
        self.eos_coef = eos_coef
        self.empty_weight = paddle.ones((self.num_classes + 1, ))
        self.empty_weight[-1] = self.eos_coef

        # Pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, sample_dict, net_out_dict, indices, num_masks):
        src_logits = net_out_dict['logits'].astype('float32')

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = paddle.concat(
            [t[J] for t, (_, J) in zip(sample_dict['gt_ids'], indices)])
        target_classes = paddle.full(
            src_logits.shape[:2], self.num_classes, dtype='int64')
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose((0, 2, 1)),
            target_classes,
            self.empty_weight,
            axis=1)
        return self.weight_ce * loss_ce

    def loss_masks(self, sample_dict, net_out_dict, indices, num_masks):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = net_out_dict['masks']
        src_masks = src_masks[src_idx]
        if src_masks.dim() == 2:
            src_masks = src_masks.unsqueeze(0)
        masks = sample_dict['gt_masks']
        target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks[tgt_idx]
        if target_masks.dim() == 2:
            target_masks = target_masks.unsqueeze(0)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with paddle.no_grad():
            # Sample point_coords
            point_coords = d2_get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio, )
            # Get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False, ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False, ).squeeze(1)

        loss_ce = sigmoid_ce_loss(point_logits, point_labels, num_masks)
        loss_dice = dice_loss(point_logits, point_labels, num_masks)
        return self.weight_mask * loss_ce + self.weight_dice * loss_dice

    def _get_src_permutation_idx(self, indices):
        # Permute predictions following indices
        batch_idx = paddle.concat(
            [paddle.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = paddle.concat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # Permute targets following indices
        batch_idx = paddle.concat(
            [paddle.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = paddle.concat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def compute_losses(self, net_out, data):
        # Check if 'gt_ids' and 'gt_masks' exist and convert the values to tensor
        if 'gt_ids' not in data:
            raise ValueError("Sample dict does not contain the key 'gt_ids'.")
        if 'gt_masks' not in data:
            raise ValueError("Sample dict does not contain the key 'gt_masks'.")
        nonzero_len_idcs = [
            i for i, gt_ids in enumerate(data['gt_ids']) if len(gt_ids) > 0
        ]
        nonzero_len_idcs = paddle.to_tensor(nonzero_len_idcs, dtype='int64')
        # If there are no gt targets, return zero loss.
        if len(nonzero_len_idcs) == 0:
            return (data['logits'] - data['logits']).mean()
        # XXX: Inplace modification
        data['gt_ids'] = [
            paddle.to_tensor(
                gt_ids, dtype='int64')
            for i, gt_ids in enumerate(data['gt_ids']) if i in nonzero_len_idcs
        ]
        data['gt_masks'] = [
            paddle.to_tensor(
                gt_masks, dtype='float32')
            for i, gt_masks in enumerate(data['gt_masks'])
            if i in nonzero_len_idcs
        ]
        if len(nonzero_len_idcs) < net_out['logits'].shape[0]:
            net_out['logits'] = paddle.index_select(
                net_out['logits'], nonzero_len_idcs, axis=0)
            net_out['masks'] = paddle.index_select(
                net_out['masks'], nonzero_len_idcs, axis=0)

        # Retrieve the matching between the outputs and the targets
        indices = self.matcher(data, net_out)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_masks = sum(len(gt_ids) for gt_ids in data['gt_ids'])
        num_masks = paddle.to_tensor([num_masks], dtype='float32')
        n_ranks = paddle.distributed.get_world_size()
        if n_ranks > 1:
            paddle.distributed.all_reduce(num_masks)
        num_masks = int(paddle.clip(num_masks / n_ranks, min=1))

        l1 = self.loss_labels(data, net_out, indices, num_masks)
        l2 = self.loss_masks(data, net_out, indices, num_masks)
        loss = l1 + l2

        # Calculate auxiliary losses
        if 'aux_masks' in net_out and 'aux_logits' in net_out:
            for i, (
                    aux_logit, aux_mask
            ) in enumerate(zip(net_out['aux_logits'], net_out['aux_masks'])):
                aux_dict = build_info_dict(
                    _type_='net_out', logits=aux_logit, masks=aux_mask)
                if len(nonzero_len_idcs) < aux_dict['logits'].shape[0]:
                    aux_dict['logits'] = paddle.index_select(
                        aux_dict['logits'], nonzero_len_idcs, axis=0)
                    aux_dict['masks'] = paddle.index_select(
                        aux_dict['masks'], nonzero_len_idcs, axis=0)
                indices = self.matcher(data, aux_dict)
                l1 = self.loss_labels(data, aux_dict, indices, num_masks)
                l2 = self.loss_masks(data, aux_dict, indices, num_masks)
                loss += l1 + l2

        return [loss]

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os

import numpy as np
import paddle
import paddle.nn.functional as F

from paddleseg.utils import metrics, Timer, calculate_eta, logger, progbar

np.set_printoptions(suppress=True)


def get_reverse_list(ori_label, transforms):
    """
    get reverse list of transform.

    Args:
        ori_label (Tensor): Origin label
        transforms (List): List of transform.

    Returns:
        list: List of tuple, there are two format:
            ('resize', (h, w)) The image shape before resize,
            ('padding', (h, w)) The image shape before padding.
    """
    reverse_list = []
    h, w = ori_label.shape[-2], ori_label.shape[-1]
    for op in transforms:
        if op.__class__.__name__ in ['Resize', 'ResizeByLong']:
            reverse_list.append(('resize', (h, w)))
            h, w = op.target_size[0], op.target_size[1]
        if op.__class__.__name__ in ['Padding']:
            reverse_list.append(('padding', (h, w)))
            w, h = op.target_size[0], op.target_size[1]
    return reverse_list


def reverse_transform(pred, ori_label, transforms):
    """recover pred to origin shape"""
    reverse_list = get_reverse_list(ori_label, transforms)
    for item in reverse_list[::-1]:
        if item[0] == 'resize':
            h, w = item[1][0], item[1][1]
            pred = F.interpolate(pred, (h, w), mode='nearest')
        elif item[0] == 'padding':
            h, w = item[1][0], item[1][1]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return pred


def evaluate(model, eval_dataset=None, iter_id=None, num_workers=0):
    model.eval()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()
    batch_sampler = paddle.io.DistributedBatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )

    total_iters = len(loader)
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0

    logger.info("Start evaluating (total_samples={}, total_iters={})...".format(
        len(eval_dataset), total_iters))
    progbar_val = progbar.Progbar(target=total_iters, verbose=1)
    for iter, (im, label) in enumerate(loader):
        label = label.astype('int64')

        logits = model(im)
        pred = logits[0]
        #         pred = paddle.argmax(pred, axis=1, keepdim=True, dtype='int32')
        pred = paddle.argmax(pred, axis=0, keepdim=True, dtype='int32')
        pred = reverse_transform(pred, label,
                                 eval_dataset.transforms.transforms)

        intersect_area, pred_area, label_area = metrics.calculate_area(
            pred,
            label,
            eval_dataset.num_classes,
            ignore_index=eval_dataset.ignore_index)

        # Gather from all ranks
        if nranks > 1:
            intersect_area_list = []
            pred_area_list = []
            label_area_list = []
            paddle.distributed.all_gather(intersect_area_list, intersect_area)
            paddle.distributed.all_gather(pred_area_list, pred_area)
            paddle.distributed.all_gather(label_area_list, label_area)

            # Some image has been evaluated and should be eliminated in last iter
            if (iter + 1) * nranks > len(eval_dataset):
                valid = len(eval_dataset) - iter * nranks
                intersect_area_list = intersect_area_list[:valid]
                pred_area_list = pred_area_list[:valid]
                label_area_list = label_area_list[:valid]

            for i in range(len(intersect_area_list)):
                intersect_area_all = intersect_area_all + intersect_area_list[i]
                pred_area_all = pred_area_all + pred_area_list[i]
                label_area_all = label_area_all + label_area_list[i]
        else:
            intersect_area_all = intersect_area_all + intersect_area
            pred_area_all = pred_area_all + pred_area
            label_area_all = label_area_all + label_area

        if local_rank == 0:
            progbar_val.update(iter + 1)

    category_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                                          label_area_all)
    category_acc, acc = metrics.accuracy(intersect_area_all, pred_area_all)
    kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)

    logger.info("[EVAL] #Images={} mIoU={:.4f} Acc={:.4f} Kappa={:.4f} ".format(
        len(eval_dataset), miou, acc, kappa))
    logger.info("[EVAL] Category IoU: \n" + str(np.round(category_iou, 4)))
    logger.info("[EVAL] Category Acc: \n" + str(np.round(category_acc, 4)))
    return miou, acc

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

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
import tqdm

from paddleseg.utils import ConfusionMatrix, Timer, calculate_eta, logger

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
    """recover to origin shape"""
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
        # Initialize parallel training environment.
        paddle.distributed.init_parallel_env()
        strategy = paddle.distributed.prepare_context()
        ddp_model = paddle.DataParallel(model, strategy)
    batch_sampler = paddle.io.DistributedBatchSampler(
        eval_dataset, batch_size=4, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )

    total_iters = len(loader)
    conf_mat = ConfusionMatrix(eval_dataset.num_classes, streaming=True)

    logger.info("Start evaluating (total_samples={}, total_iters={})...".format(
        len(eval_dataset), total_iters))
    timer = Timer()
    timer.start()
    reader_time = 0
    logit_time = 0
    gather_time = 0
    tocpu_time = 0
    conf_time = 0
    all_time = 0
    for iter, (im, label) in enumerate(loader):
        reader_cost = timer.elapsed_time()
        reader_time += reader_cost
        label = label.astype('int64')

        logit_start = timer.elapsed_time()
        if nranks > 1:
            logits = ddp_model(im)
        else:
            logits = model(im)
        pred = logits[0]
        pred = paddle.argmax(pred, axis=1)
        pred = reverse_transform(pred, label,
                                 eval_dataset.transforms.transforms)
        if local_rank == 0:
            logit_time += timer.elapsed_time() - logit_start

        # Gather and concat pred and label from all ranks
        gather_start = timer.elapsed_time()
        if nranks > 1:
            pred_list = []
            label_list = []
            paddle.distributed.all_gather(pred_list, pred)
            paddle.distributed.all_gather(label_list, label)

            # Some image has been evaluated and should be eliminated in last iter
            if (iter + 1) * nranks > len(eval_dataset):
                valid = len(eval_dataset) - iter * nranks
                pred_list = pred_list[:valid]
                label_list = label_list[:valid]
            pred = paddle.concat(pred_list, axis=0)
            label = paddle.concat(label_list, axis=0)
        if local_rank == 0:
            gather_time += timer.elapsed_time() - gather_start

        tocpu_start = timer.elapsed_time()
        pred = pred.numpy()
        label = label.numpy()
        if local_rank == 0:
            tocpu_time += timer.elapsed_time() - tocpu_start
        mask = label != eval_dataset.ignore_index

        conf_start = timer.elapsed_time()
        conf_mat.calculate(pred=pred, label=label, mask=mask)
        _, iou = conf_mat.mean_iou()
        if local_rank == 0:
            conf_time += timer.elapsed_time() - conf_start

        batch_cost = timer.elapsed_time()
        all_time += batch_cost
        remain_iter = total_iters - iter - 1
        logger.info(
            "[EVAL] iter_id={}, iter={}/{}, IoU={:4f}, batch_cost={:.4f}, reader_cost={:4f} | ETA {}"
            .format(iter_id, iter + 1, total_iters, iou, batch_cost,
                    reader_cost, calculate_eta(remain_iter, batch_cost)))
        timer.restart()

    category_iou, miou = conf_mat.mean_iou()
    category_acc, acc = conf_mat.accuracy()
    logger.info("[EVAL] #Images={} mIoU={:.4f} Acc={:.4f} Kappa={:.4f} ".format(
        len(eval_dataset), miou, acc, conf_mat.kappa()))
    logger.info("[EVAL] Category IoU: \n" + str(np.round(category_iou, 4)))
    logger.info("[EVAL] Category Acc: \n" + str(np.round(category_acc, 4)))
    logger.info(
        'reader_time: {},\nlogit_time: {},\ngather_time: {},\n tocpu_time: {}\n, conf_time: {},\n all_time: {}'
        .format(reader_time, logit_time, gather_time, tocpu_time, conf_time,
                all_time))
    return miou, acc

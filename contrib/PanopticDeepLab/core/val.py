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

import os
from collections import OrderedDict

import numpy as np
import time
import paddle
import paddle.nn.functional as F
from paddleseg.utils import TimeAverager, calculate_eta, logger, progbar

from utils.evaluation import SemanticEvaluator, InstanceEvaluator, PanopticEvaluator
from core import infer

np.set_printoptions(suppress=True)


def evaluate(model,
             eval_dataset,
             threshold=0.1,
             nms_kernel=7,
             top_k=200,
             num_workers=0,
             print_detail=True):
    """
    Launch evaluation.

    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        threshold (float, optional): Threshold applied to center heatmap score. Defalut: 0.1.
        nms_kernel (int, optional): NMS max pooling kernel size. Default: 7.
        top_k (int, optional): Top k centers to keep. Default: 200.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.

    Returns:
        dict: Panoptic evaluation results which includes PQ, RQ, SQ for all, each class, Things and stuff.
        dict: Semantic evaluation results which includes mIoU, fwIoU, mACC and pACC.
        dict: Instance evaluation results which includes mAP and mAP50, and also AP and AP50 for each class.

    """
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
    semantic_metric = SemanticEvaluator(
        eval_dataset.num_classes, ignore_index=eval_dataset.ignore_index)
    instance_metric_AP50 = InstanceEvaluator(
        eval_dataset.num_classes,
        overlaps=0.5,
        thing_list=eval_dataset.thing_list)
    instance_metric_AP = InstanceEvaluator(
        eval_dataset.num_classes,
        overlaps=list(np.arange(0.5, 1.0, 0.05)),
        thing_list=eval_dataset.thing_list)
    panoptic_metric = PanopticEvaluator(
        num_classes=eval_dataset.num_classes,
        thing_list=eval_dataset.thing_list,
        ignore_index=eval_dataset.ignore_index,
        label_divisor=eval_dataset.label_divisor)

    if print_detail:
        logger.info(
            "Start evaluating (total_samples={}, total_iters={})...".format(
                len(eval_dataset), total_iters))
    progbar_val = progbar.Progbar(target=total_iters, verbose=1)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    with paddle.no_grad():
        for iter, data in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)
            im = data[0]
            raw_semantic_label = data[1]  # raw semantic label.
            raw_instance_label = data[2]
            raw_panoptic_label = data[3]
            ori_shape = raw_semantic_label.shape[-2:]

            semantic, semantic_softmax, instance, panoptic, ctr_hmp = infer.inference(
                model=model,
                im=im,
                transforms=eval_dataset.transforms.transforms,
                thing_list=eval_dataset.thing_list,
                label_divisor=eval_dataset.label_divisor,
                stuff_area=eval_dataset.stuff_area,
                ignore_index=eval_dataset.ignore_index,
                threshold=threshold,
                nms_kernel=nms_kernel,
                top_k=top_k,
                ori_shape=ori_shape)
            semantic = semantic.squeeze().numpy()
            semantic_softmax = semantic_softmax.squeeze().numpy()
            instance = instance.squeeze().numpy()
            panoptic = panoptic.squeeze().numpy()
            ctr_hmp = ctr_hmp.squeeze().numpy()
            raw_semantic_label = raw_semantic_label.squeeze().numpy()
            raw_instance_label = raw_instance_label.squeeze().numpy()
            raw_panoptic_label = raw_panoptic_label.squeeze().numpy()

            # update metric for semantic, instance, panoptic
            semantic_metric.update(semantic, raw_semantic_label)

            gts = instance_metric_AP.convert_gt_map(raw_semantic_label,
                                                    raw_instance_label)
            # print([i[0] for i in gts])
            preds = instance_metric_AP.convert_pred_map(semantic_softmax,
                                                        panoptic)
            # print([(i[0], i[1]) for i in preds ])
            ignore_mask = raw_semantic_label == eval_dataset.ignore_index
            instance_metric_AP.update(preds, gts, ignore_mask=ignore_mask)
            instance_metric_AP50.update(preds, gts, ignore_mask=ignore_mask)

            panoptic_metric.update(panoptic, raw_panoptic_label)

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(im))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if local_rank == 0:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                              ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    semantic_results = semantic_metric.evaluate()
    panoptic_results = panoptic_metric.evaluate()
    instance_results = OrderedDict()
    ins_ap = instance_metric_AP.evaluate()
    ins_ap50 = instance_metric_AP50.evaluate()
    instance_results['ins_seg'] = OrderedDict()
    instance_results['ins_seg']['mAP'] = ins_ap['ins_seg']['mAP']
    instance_results['ins_seg']['AP'] = ins_ap['ins_seg']['AP']
    instance_results['ins_seg']['mAP50'] = ins_ap50['ins_seg']['mAP']
    instance_results['ins_seg']['AP50'] = ins_ap50['ins_seg']['AP']

    if print_detail:
        logger.info(panoptic_results)
        print()
        logger.info(semantic_results)
        print()
        logger.info(instance_results)
        print()

        pq = panoptic_results['pan_seg']['All']['pq']
        miou = semantic_results['sem_seg']['mIoU']
        map = instance_results['ins_seg']['mAP']
        map50 = instance_results['ins_seg']['mAP50']
        logger.info(
            "PQ: {:.4f}, mIoU: {:.4f}, mAP: {:.4f}, mAP50: {:.4f}".format(
                pq, miou, map, map50))

    return panoptic_results, semantic_results, instance_results

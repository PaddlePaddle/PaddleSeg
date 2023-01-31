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

from collections import OrderedDict

import numpy as np
import time
import paddle
from paddleseg.utils import TimeAverager, logger, progbar

from paddlepanseg.cvlibs import build_info_dict
from paddlepanseg.utils.evaluation import SemSegEvaluator, InsSegEvaluator, PanSegEvaluator
from paddlepanseg.utils import tabulate_metrics
from paddlepanseg.core.runner import PanSegRunner
from paddlepanseg.core.launcher import AMPLauncher

np.set_printoptions(suppress=True)


def evaluate(model,
             eval_dataset,
             postprocessor,
             num_workers=0,
             print_detail=True,
             eval_sem=False,
             eval_ins=False,
             precision='fp32',
             amp_level='O1',
             runner=None):
    """
    Launch evaluation.

    Args:
        model（nn.Layer): A panoptic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        postprocessor (paddlepanseg.postprocessors.Postprocessor): Used to postprocess model output.
        runner (paddlepanseg.core.runners.PanSegRunner): Used to define how the components interact.
        num_workers (int, optional): The number of workers used by the data loader. Default: 0.
        print_detail (bool, optional): Whether or not to print detailed information about the evaluation process. Default: True.
        eval_sem (bool, optional): Whether or not to calculate semantic segmentation metrics. Default: False.
        eval_ins (bool, optional): Whether or not to calculate instance segmentation metrics. Default: False.
        precision (str, optional): If `precision` is 'fp16', enable automatic mixed precision training. Default: 'fp32'.
        amp_level (str, optional): The auto mixed precision level. Choices are 'O1' and 'O2'. Default: 'O1'.
    """

    model.eval()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()

    # By default evaluate the results with panoptic segmentation metrics
    eval_pan = True

    # Build batch sampler and data loader
    batch_sampler = paddle.io.DistributedBatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
        collate_fn=eval_dataset.collate
        if hasattr(eval_dataset, 'collate') else None)

    # Bind components to runner
    runner.bind(model=model, postprocessor=postprocessor)

    # Create launcher
    launcher = AMPLauncher(
        runner=runner, precision=precision, amp_level=amp_level)

    total_iters = len(loader)
    if eval_sem:
        sem_evaluator = SemSegEvaluator(
            eval_dataset.num_classes, ignore_index=eval_dataset.ignore_index)
    if eval_ins:
        ins_evaluator_ap50 = InsSegEvaluator(
            eval_dataset.num_classes,
            overlaps=0.5,
            thing_ids=eval_dataset.thing_ids,
            ignore_index=eval_dataset.ignore_index)
        ins_evaluator_ap = InsSegEvaluator(
            eval_dataset.num_classes,
            overlaps=list(np.arange(0.5, 1.0, 0.05)),
            thing_ids=eval_dataset.thing_ids,
            ignore_index=eval_dataset.ignore_index)
    if eval_pan:
        pan_evaluator = PanSegEvaluator(
            num_classes=eval_dataset.num_classes,
            thing_ids=eval_dataset.thing_ids,
            label_divisor=eval_dataset.label_divisor,
            convert_id=eval_dataset.convert_id_for_eval)

    if print_detail:
        logger.info("Start evaluating (total_samples={}, total_iters={})...".
                    format(len(eval_dataset), total_iters))
    progbar_val = progbar.Progbar(target=total_iters, verbose=1)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    with paddle.no_grad():
        for iter, data in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)

            pp_out = launcher.infer_step(data=data)

            if eval_sem:
                sem_evaluator.update(data, pp_out)
            if eval_ins:
                ins_evaluator_ap.update(data, pp_out)
                ins_evaluator_ap50.update(data, pp_out)
            if eval_pan:
                pan_evaluator.update(data, pp_out)

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(data['img']))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if local_rank == 0:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                              ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    results = build_info_dict(_type_='metric')

    if eval_sem:
        sem_results = sem_evaluator.evaluate()
        results['sem_metrics'] = sem_results['sem_metrics']
    if eval_ins:
        ins_results = build_info_dict(
            _type_='metric', ins_metrics=OrderedDict())
        ins_results_ap = ins_evaluator_ap.evaluate()
        ins_results_ap50 = ins_evaluator_ap50.evaluate()
        ins_results['ins_metrics']['mAP'] = ins_results_ap['ins_metrics']['mAP']
        ins_results['ins_metrics']['AP'] = ins_results_ap['ins_metrics']['AP']
        ins_results['ins_metrics']['mAP50'] = ins_results_ap50['ins_metrics'][
            'mAP']
        ins_results['ins_metrics']['AP50'] = ins_results_ap50['ins_metrics'][
            'AP']
        results['ins_metrics'] = ins_results['ins_metrics']
    if eval_pan:
        pan_results = pan_evaluator.evaluate()
        results['pan_metrics'] = pan_results['pan_metrics']

    if print_detail:
        if eval_pan:
            logger.info(
                tabulate_metrics(
                    results['pan_metrics']['All'],
                    digits=4,
                    title="Pan Metrics of All Classes:"))
            logger.info(
                tabulate_metrics(
                    results['pan_metrics']['Things'],
                    digits=4,
                    title="Pan Metrics of Thing Classes:"))
            logger.info(
                tabulate_metrics(
                    results['pan_metrics']['Stuff'],
                    digits=4,
                    title="Pan Metrics of Stuff Classes:"))
            logger.info("PQ: {:.4f}".format(results['pan_metrics']['All'][
                'pq']))
        if eval_sem:
            logger.info(
                tabulate_metrics(
                    results['sem_metrics'], digits=2, title="Sem metrics:"))
            logger.info("mIoU: {:.4f}".format(results['sem_metrics']['mIoU']))
        if eval_ins:
            logger.info(
                tabulate_metrics(
                    results['ins_metrics'], digits=2, title="Ins metrics:"))
            logger.info("mAP: {:.4f}, mAP50: {:.4f}".format(results[
                'ins_metrics']['mAP'], results['ins_metrics']['mAP50']))

    return results

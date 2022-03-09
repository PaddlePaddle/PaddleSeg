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
import time
import paddle
import paddle.nn.functional as F

from paddleseg.utils import metrics, TimeAverager, calculate_eta, logger, progbar
from . import infer
from third_party import tusimple_processor

np.set_printoptions(suppress=True)


def evaluate(model,
             eval_dataset,
             num_workers=0,
             is_view=False,
             save_dir='output',
             print_detail=True):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        is_view (bool, optional): Whether to visualize results. Default: False.
        save_dir (str, optional): The directory to save the json or visualized results. Default: 'output'.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.

    Returns:
        float: The acc of validation datasets.
        float: The fp of validation datasets.
        float: The fn of validation datasets.
    """
    model.eval()
    local_rank = paddle.distributed.ParallelEnv().local_rank

    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_size=4,
        drop_last=False,
        num_workers=num_workers,
        return_list=True,
    )

    postprocessor = tusimple_processor.TusimpleProcessor(
        num_classes=eval_dataset.num_classes,
        cut_height=eval_dataset.cut_height,
        test_gt_json=eval_dataset.test_gt_json,
        save_dir=save_dir,
    )
    total_iters = len(loader)

    if print_detail:
        logger.info(
            "Start evaluating (total_samples: {}, total_iters: {})...".format(
                len(eval_dataset), total_iters))
    progbar_val = progbar.Progbar(target=total_iters, verbose=1)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    with paddle.no_grad():
        for iter, (im, label, im_path) in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)
            label = label.astype('int64')

            ori_shape = None

            time_start = time.time()
            pred = infer.inference(
                model,
                im,
                ori_shape=ori_shape,
                transforms=eval_dataset.transforms.transforms)
            time_end = time.time()

            postprocessor.dump_data_to_json(
                pred[1],
                im_path,
                run_time=time_end - time_start,
                is_dump_json=True,
                is_view=is_view)
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if local_rank == 0 and print_detail:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                              ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    acc, fp, fn, eval_result = postprocessor.bench_one_submit(local_rank)

    if print_detail:
        logger.info(eval_result)
    return acc, fp, fn

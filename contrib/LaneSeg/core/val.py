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

import numpy as np
import time
import paddle
from tqdm import tqdm
from paddleseg.utils import metrics, TimeAverager, calculate_eta, logger, progbar
from . import infer
from utils import tusimple

np.set_printoptions(suppress=True)


def evaluate(model, eval_dataset, num_workers=0, print_detail=True):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
        float: The lane acc of validation datasets.
        float: The lane fn of validation datasets.
        float: The lane fp of validation datasets.
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
        eval_dataset, batch_size=4, shuffle=False, drop_last=False)

    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )

    postprocessor = tusimple.Tusimple(
        num_classes=eval_dataset.num_classes,
        cut_height=eval_dataset.cut_height,
        test_gt_json=eval_dataset.test_gt_json,
    )

    total_iters = len(loader)

    if print_detail:
        logger.info(
            "Start evaluating (total_samples: {}, total_iters: {})...".format(
                len(eval_dataset), total_iters))
    # TODO(chenguowei): fix log print error with multi-gpus
    progbar_val = progbar.Progbar(
        target=total_iters, verbose=1 if nranks < 2 else 2)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    with paddle.no_grad():
        for i, (im, label, im_path) in enumerate(
                tqdm(loader, desc=f'Validate')):
            reader_cost_averager.record(time.time() - batch_start)
            label = label.astype('int64')
            # For lane tasks, use the post-processed size when evaluating
            ori_shape = label.shape[-2:]

            pred = infer.inference(
                model,
                im,
                ori_shape=ori_shape,
                transforms=eval_dataset.transforms.transforms)

            postprocessor.evaluate(pred[1], im_path)
        acc, fp, fn, eval_result = postprocessor.calculate_eval()
        logger.info(eval_result)
    return acc, fn, fp

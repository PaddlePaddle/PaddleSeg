# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import time
import numpy as np

import paddle
import paddle.nn.functional as F

from medicalseg.utils import metric, TimeAverager, calculate_eta, logger, progbar, loss_computation, add_image_vdl
from nnunet.utils import sum_tensor
np.set_printoptions(suppress=True)


def evaluate(model,
             eval_dataset,
             losses,
             num_workers=0,
             print_detail=True,
             auc_roc=False,
             writer=None,
             save_dir=None):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        losses(dict): Used to calculate the loss. e.g: {"types":[loss_1...], "coef": [0.5,...]}
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.
        auc_roc(bool, optional): whether add auc_roc metric.
        writer: visualdl log writer.
        save_dir(str, optional): the path to save predicted result.

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()
    loader = eval_dataset
    total_iters = len(loader)

    if print_detail:
        logger.info("Start evaluating (total_samples: {}, total_iters: {})...".
                    format(len(eval_dataset), total_iters))
    progbar_val = progbar.Progbar(target=total_iters, verbose=1)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()

    online_eval_foreground_dc = []
    online_eval_tp_list = []
    online_eval_fp_list = []
    online_eval_fn_list = []

    with paddle.no_grad():
        for iter, (im, label) in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)

            label = label[0].astype('int32')

            if iter >= eval_dataset.num_batches_per_epoch:
                break
            output = model(im)[0]
            num_classes = output.shape[1]

            output_softmax = F.softmax(output, axis=1)
            output_seg = paddle.argmax(output_softmax, axis=1)

            target = label[:, 0, ...]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = paddle.zeros((target.shape[0], num_classes - 1))
            fp_hard = paddle.zeros((target.shape[0], num_classes - 1))
            fn_hard = paddle.zeros((target.shape[0], num_classes - 1))
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor(
                    (output_seg == c).astype('float32') *
                    (target == c).astype('float32'),
                    axes=axes)
                fp_hard[:, c - 1] = sum_tensor(
                    (output_seg == c).astype('float32') *
                    (target != c).astype('float32'),
                    axes=axes)
                fn_hard[:, c - 1] = sum_tensor(
                    (output_seg != c).astype('float32') *
                    (target == c).astype('float32'),
                    axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).numpy()

            online_eval_foreground_dc.append(
                list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            online_eval_tp_list.append(list(tp_hard))
            online_eval_fp_list.append(list(fp_hard))
            online_eval_fn_list.append(list(fn_hard))

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if print_detail:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                              ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    online_eval_tp = np.sum(online_eval_tp_list, 0)
    online_eval_fp = np.sum(online_eval_fp_list, 0)
    online_eval_fn = np.sum(online_eval_fn_list, 0)

    global_dc_per_class = [
        i
        for i in [
            2 * i / (2 * i + j + k)
            for i, j, k in zip(online_eval_tp, online_eval_fp, online_eval_fn)
        ] if not np.isnan(i)
    ]
    mdice = np.mean(global_dc_per_class)

    logger.info("Average global foreground Dice: {}".format(
        [np.round(i, 4) for i in global_dc_per_class]))
    logger.info(
        "(interpret this as an estimate for the Dice of the different classes. This is not "
        "exact.)")
    result_dict = {"mdice": mdice}
    return result_dict

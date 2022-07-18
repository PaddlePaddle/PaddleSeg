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
import time
import paddle
import paddle.nn.functional as F
from paddleseg.utils import TimeAverager, calculate_eta, logger, progbar

from ppmatting.metrics import metrics_class_dict

np.set_printoptions(suppress=True)


def save_alpha_pred(alpha, path):
    """
    The value of alpha is range [0, 1], shape should be [h,w]
    """
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    alpha = (alpha).astype('uint8')
    cv2.imwrite(path, alpha)


def reverse_transform(alpha, trans_info):
    """recover pred to origin shape"""
    for item in trans_info[::-1]:
        if item[0][0] == 'resize':
            h, w = item[1][0], item[1][1]
            alpha = F.interpolate(alpha, [h, w], mode='bilinear')
        elif item[0][0] == 'padding':
            h, w = item[1][0], item[1][1]
            alpha = alpha[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return alpha


def evaluate(model,
             eval_dataset,
             num_workers=0,
             print_detail=True,
             save_dir='output/results',
             save_results=True,
             metrics='sad'):
    model.eval()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()

    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_size=1,
        drop_last=False,
        num_workers=num_workers,
        return_list=True, )

    total_iters = len(loader)
    # Get metric instances and data saving
    metrics_ins = {}
    metrics_data = {}
    if isinstance(metrics, str):
        metrics = [metrics]
    elif not isinstance(metrics, list):
        metrics = ['sad']
    for key in metrics:
        key = key.lower()
        metrics_ins[key] = metrics_class_dict[key]()
        metrics_data[key] = None

    if print_detail:
        logger.info("Start evaluating (total_samples: {}, total_iters: {})...".
                    format(len(eval_dataset), total_iters))
    progbar_val = progbar.Progbar(
        target=total_iters, verbose=1 if nranks < 2 else 2)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()

    img_name = ''
    i = 0
    with paddle.no_grad():
        for iter, data in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)
            alpha_pred = model(data)

            alpha_pred = reverse_transform(alpha_pred, data['trans_info'])
            alpha_pred = alpha_pred.numpy()

            alpha_gt = data['alpha'].numpy() * 255
            trimap = data.get('ori_trimap')
            if trimap is not None:
                trimap = trimap.numpy().astype('uint8')
            alpha_pred = np.round(alpha_pred * 255)
            for key in metrics_ins.keys():
                metrics_data[key] = metrics_ins[key].update(alpha_pred,
                                                            alpha_gt, trimap)

            if save_results:
                alpha_pred_one = alpha_pred[0].squeeze()
                if trimap is not None:
                    trimap = trimap.squeeze().astype('uint8')
                    alpha_pred_one[trimap == 255] = 255
                    alpha_pred_one[trimap == 0] = 0

                save_name = data['img_name'][0]
                name, ext = os.path.splitext(save_name)
                if save_name == img_name:
                    save_name = name + '_' + str(i) + ext
                    i += 1
                else:
                    img_name = save_name
                    save_name = name + '_' + str(i) + ext
                    i = 1

                save_alpha_pred(alpha_pred_one,
                                os.path.join(save_dir, save_name))

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(alpha_gt))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if local_rank == 0 and print_detail:
                show_list = [(k, v) for k, v in metrics_data.items()]
                show_list = show_list + [('batch_cost', batch_cost),
                                         ('reader cost', reader_cost)]
                progbar_val.update(iter + 1, show_list)

            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    for key in metrics_ins.keys():
        metrics_data[key] = metrics_ins[key].evaluate()
    log_str = '[EVAL] '
    for key, value in metrics_data.items():
        log_str = log_str + key + ': {:.4f}, '.format(value)
    log_str = log_str[:-2]

    logger.info(log_str)
    return metrics_data

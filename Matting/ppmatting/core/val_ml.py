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

from ppmatting.metrics import metric
from pymatting.util.util import load_image, save_image, stack_images
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml

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
            h, w = int(item[1][0]), int(item[1][1])
            alpha = cv2.resize(alpha, dsize=(w, h))
        elif item[0][0] == 'padding':
            h, w = int(item[1][0]), int(item[1][1])
            alpha = alpha[0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return alpha


def evaluate_ml(model,
                eval_dataset,
                num_workers=0,
                print_detail=True,
                save_dir='output/results',
                save_results=True):

    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_size=1,
        drop_last=False,
        num_workers=num_workers,
        return_list=True, )

    total_iters = len(loader)
    mse_metric = metric.MSE()
    sad_metric = metric.SAD()
    grad_metric = metric.Grad()
    conn_metric = metric.Conn()

    if print_detail:
        logger.info("Start evaluating (total_samples: {}, total_iters: {})...".
                    format(len(eval_dataset), total_iters))
    progbar_val = progbar.Progbar(target=total_iters, verbose=1)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()

    img_name = ''
    i = 0
    ignore_cnt = 0
    for iter, data in enumerate(loader):

        reader_cost_averager.record(time.time() - batch_start)

        image_rgb_chw = data['img'].numpy()[0]
        image_rgb_hwc = np.transpose(image_rgb_chw, (1, 2, 0))
        trimap = data['trimap'].numpy().squeeze() / 255.0
        image = image_rgb_hwc * 0.5 + 0.5  # reverse normalize (x/255 - mean) / std

        is_fg = trimap >= 0.9
        is_bg = trimap <= 0.1

        if is_fg.sum() == 0 or is_bg.sum() == 0:
            ignore_cnt += 1
            logger.info(str(iter))
            continue

        alpha_pred = model(image, trimap)

        alpha_pred = reverse_transform(alpha_pred, data['trans_info'])

        alpha_gt = data['alpha'].numpy().squeeze() * 255

        trimap = data['ori_trimap'].numpy().squeeze()

        alpha_pred = np.round(alpha_pred * 255)
        mse = mse_metric.update(alpha_pred, alpha_gt, trimap)
        sad = sad_metric.update(alpha_pred, alpha_gt, trimap)
        grad = grad_metric.update(alpha_pred, alpha_gt, trimap)
        conn = conn_metric.update(alpha_pred, alpha_gt, trimap)

        if sad > 1000:
            print(data['img_name'][0])

        if save_results:
            alpha_pred_one = alpha_pred
            alpha_pred_one[trimap == 255] = 255
            alpha_pred_one[trimap == 0] = 0

            save_name = data['img_name'][0]
            name, ext = os.path.splitext(save_name)
            if save_name == img_name:
                save_name = name + '_' + str(i) + ext
                i += 1
            else:
                img_name = save_name
                save_name = name + '_' + str(0) + ext
                i = 1
            save_alpha_pred(alpha_pred_one, os.path.join(save_dir, save_name))

        batch_cost_averager.record(
            time.time() - batch_start, num_samples=len(alpha_gt))
        batch_cost = batch_cost_averager.get_average()
        reader_cost = reader_cost_averager.get_average()

        if print_detail:
            progbar_val.update(iter + 1,
                               [('SAD', sad), ('MSE', mse), ('Grad', grad),
                                ('Conn', conn), ('batch_cost', batch_cost),
                                ('reader cost', reader_cost)])

        reader_cost_averager.reset()
        batch_cost_averager.reset()
        batch_start = time.time()

    mse = mse_metric.evaluate()
    sad = sad_metric.evaluate()
    grad = grad_metric.evaluate()
    conn = conn_metric.evaluate()

    logger.info('[EVAL] SAD: {:.4f}, MSE: {:.4f}, Grad: {:.4f}, Conn: {:.4f}'.
                format(sad, mse, grad, conn))
    logger.info('{}'.format(ignore_cnt))

    return sad, mse, grad, conn

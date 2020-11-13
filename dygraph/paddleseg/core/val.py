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
import tqdm

from paddleseg.utils import ConfusionMatrix, Timer, calculate_eta, logger

np.set_printoptions(suppress=True)


def evaluate(model, eval_dataset=None, iter_id=None):
    model.eval()

    total_iters = len(eval_dataset)
    conf_mat = ConfusionMatrix(eval_dataset.num_classes, streaming=True)

    logger.info("Start evaluating (total_samples={}, total_iters={})...".format(
        len(eval_dataset), total_iters))
    timer = Timer()
    timer.start()
    for iter, (im, im_info, label) in tqdm.tqdm(
            enumerate(eval_dataset), total=total_iters):
        im = paddle.to_tensor(im)
        logits = model(im)
        pred = paddle.argmax(logits[0], axis=1)
        pred = pred.numpy().astype('float32')
        pred = np.squeeze(pred)
        for info in im_info[::-1]:
            if info[0] == 'resize':
                h, w = info[1][0], info[1][1]
                pred = cv2.resize(pred, (w, h), cv2.INTER_NEAREST)
            elif info[0] == 'padding':
                h, w = info[1][0], info[1][1]
                pred = pred[0:h, 0:w]
            else:
                raise ValueError("Unexpected info '{}' in im_info".format(
                    info[0]))
        pred = pred[np.newaxis, :, :, np.newaxis]
        pred = pred.astype('int64')
        mask = label != eval_dataset.ignore_index
        # To-DO Test Execution Time
        conf_mat.calculate(pred=pred, label=label, mask=mask)
        _, iou = conf_mat.mean_iou()

        time_iter = timer.elapsed_time()
        remain_iter = total_iters - iter - 1
        logger.debug(
            "[EVAL] iter_id={}, iter={}/{}, IoU={:4f}, sec/iter={:.4f} | ETA {}"
            .format(iter_id, iter + 1, total_iters, iou, time_iter,
                    calculate_eta(remain_iter, time_iter)))
        timer.restart()

    class_iou, miou = conf_mat.mean_iou()
    class_acc, acc = conf_mat.accuracy()
    logger.info("[EVAL] #Images={} mIoU={:.4f} Acc={:.4f} Kappa={:.4f} ".format(
        len(eval_dataset), miou, acc, conf_mat.kappa()))
    logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
    logger.info("[EVAL] Class Acc: \n" + str(np.round(class_acc, 4)))
    return miou, acc

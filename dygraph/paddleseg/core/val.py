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
import tqdm
import cv2
from paddle.fluid.dygraph.base import to_variable
import paddle.fluid as fluid
import paddle.nn.functional as F
import paddle

import paddleseg.utils.logger as logger
from paddleseg.utils import ConfusionMatrix
from paddleseg.utils import Timer, calculate_eta


def evaluate(model,
             eval_dataset=None,
             model_dir=None,
             num_classes=None,
             ignore_index=255,
             iter_id=None):
    ckpt_path = os.path.join(model_dir, 'model')
    para_state_dict, opti_state_dict = fluid.load_dygraph(ckpt_path)
    model.set_dict(para_state_dict)
    model.eval()

    total_iters = len(eval_dataset)
    conf_mat = ConfusionMatrix(num_classes, streaming=True)

    logger.info(
        "Start to evaluating(total_samples={}, total_iters={})...".format(
            len(eval_dataset), total_iters))
    timer = Timer()
    timer.start()
    for iter, (im, im_info, label) in tqdm.tqdm(
            enumerate(eval_dataset), total=total_iters):
        im = to_variable(im)
        # pred, _ = model(im)
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
                raise Exception("Unexpected info '{}' in im_info".format(
                    info[0]))
        pred = pred[np.newaxis, :, :, np.newaxis]
        pred = pred.astype('int64')
        mask = label != ignore_index

        conf_mat.calculate(pred=pred, label=label, ignore=mask)
        _, iou = conf_mat.mean_iou()

        time_iter = timer.elapsed_time()
        remain_iter = total_iters - iter - 1
        logger.debug(
            "[EVAL] iter_id={}, iter={}/{}, iou={:4f}, sec/iter={:.4f} | ETA {}"
            .format(iter_id, iter + 1, total_iters, iou, time_iter,
                    calculate_eta(remain_iter, time_iter)))
        timer.restart()

    category_iou, miou = conf_mat.mean_iou()
    category_acc, macc = conf_mat.accuracy()
    logger.info("[EVAL] #Images={} mAcc={:.4f} mIoU={:.4f}".format(
        len(eval_dataset), macc, miou))
    logger.info("[EVAL] Category IoU: " + str(category_iou))
    logger.info("[EVAL] Category Acc: " + str(category_acc))
    logger.info("[EVAL] Kappa:{:.4f} ".format(conf_mat.kappa()))
    return miou, macc

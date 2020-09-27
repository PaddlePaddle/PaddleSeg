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
import paddle
import paddle.nn.functional as F
from paddle import to_tensor

import paddleseg.utils.logger as logger
from paddleseg.utils import ConfusionMatrix
from paddleseg.utils import Timer, calculate_eta


def recover(pred, im_info):
    """recover to origin shape"""
    for info in im_info[::-1]:
        if info[0] == 'resize':
            h, w = info[1][0], info[1][1]
            pred = F.resize_nearest(pred, (h, w))
        elif info[0] == 'padding':
            h, w = info[1][0], info[1][1]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(info[0]))
    return pred


def slide_inference(model, im, num_classes, crop_size, stride):
    h_im, w_im = im.shape[-2:]
    w_crop, h_crop = crop_size
    w_stride, h_stride = stride
    # calculate the crop nums
    rows = np.int(np.ceil(1.0 * (h_im - h_crop) / h_stride)) + 1
    cols = np.int(np.ceil(1.0 * (w_im - w_crop) / w_stride)) + 1
    # todo 'Tensor' object does not support item assignment. If support, use tensor to calculation.
    final_logit = np.zeros([1, num_classes, h_im, w_im])
    count = np.zeros([1, 1, h_im, w_im])
    for r in range(rows):
        for c in range(cols):
            h1 = r * h_stride
            w1 = c * w_stride
            h2 = min(h1 + h_crop, h_im)
            w2 = min(w1 + w_crop, w_im)
            h1 = max(h2 - h_crop, 0)
            w1 = max(w2 - w_crop, 0)
            im_crop = im[:, :, h1:h2, w1:w2]
            im_pad = F.pad2d(im_crop, [0, h_crop, 0, w_crop])
            logit = model(im_crop)[0].numpy()
            final_logit[:, :, h1:h2, w1:w2] += logit[:, :, :h2 - h1, :w2 - w1]
            count[:, :, h1:h2, w1:w2] += 1
    if np.sum(count == 0) != 0:
        raise RuntimeError(
            'There are pixel not predicted. It is possible that stride is greater than crop_size'
        )
    final_logit = final_logit / count
    return final_logit


def inference(model, im, num_classes, is_slide, stride, crop_size):
    """
    get logit
    """
    im = np.transpose(im, (2, 0, 1))
    im = im[np.newaxis, ...]
    im = to_tensor(im)
    if not is_slide:
        logits = model(im)
        logit = logits[0]
    else:
        logit = slide_inference(
            model,
            im,
            num_classes=num_classes,
            crop_size=crop_size,
            stride=stride)
        logit = to_tensor(logit)
    return logit


def multi_scale_flip_inference(model,
                               im,
                               im_info,
                               num_classes,
                               scales=[1],
                               flip=False,
                               flip_directions=['horizontal'],
                               is_slide=False,
                               stride=None,
                               crop_size=None):
    """
    按尺度和flip逐个进行inferrence,
    对得到的loigt resize回原图进行相加

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        image (np.ndarray): the input image.
        scale (list):  image scale for resize. Default [1]
        flip (bool): whether to apply flip inference. Default True.
        flip_directions (list): flip directions, options are "horizontal" and "vertical". Default horizontal.
        model (str): inference mode, options are "whole" and "slide". Default "whole".
    """
    final_logit = 0
    h_input, w_input = im.shape[0], im.shape[1]
    for scale in scales:
        h = int(h_input * scale + 0.5)
        w = int(w_input * scale + 0.5)
        im = cv2.resize(im, (w, h))
        logit = inference(
            model,
            im,
            num_classes=num_classes,
            is_slide=is_slide,
            crop_size=crop_size,
            stride=stride)
        logit = F.resize_bilinear(logit, out_shape=(h_input, w_input))
        # im_save = paddle.argmax(logit,axis=1).numpy()[0].astype('int8')
        # cv2.imwrite('afterscale'+str(scale)+'.png', im_save*10)
        final_logit = final_logit + logit
        if flip:
            for flip_direction in flip_directions:
                if flip_direction not in ['horizontal', 'vertical']:
                    raise ValueError(
                        '`flip_directions` should be in ["horizontal", "vertical"], but it is {}'
                        .format(flip_directions))
                if flip_direction == 'horizontal':
                    im_flip = im[:, ::-1, :]
                else:
                    im_flip = im[::-1, :, :]
                logit = inference(
                    model,
                    im_flip,
                    num_classes=num_classes,
                    is_slide=is_slide,
                    crop_size=crop_size,
                    stride=stride)
                if flip_direction == 'horizontal':
                    logit = logit[:, :, :, ::-1]
                else:
                    logit = logit[:, :, ::-1, :]
                logit = F.resize_bilinear(logit, out_shape=(h_input, w_input))

                # im_save = paddle.argmax(logit,axis=1).numpy()[0].astype('int8')
                # cv2.imwrite('afterscale'+str(scale)+flip_direction+'.png', im_save*10)

                final_logit = final_logit + logit
    pred = paddle.argmax(final_logit, axis=1)
    pred = paddle.unsqueeze(pred, axis=1)
    pred = recover(pred, im_info)
    return pred


def evaluate(model,
             eval_dataset=None,
             model_dir=None,
             num_classes=None,
             scale=[1],
             flip=True,
             flip_directions=['horizontal'],
             is_slide=False,
             stride=None,
             crop_size=None,
             ignore_index=255,
             iter_id=None):
    if model_dir is None:
        raise ValueError('`model_dir` should be provided, but is None')
    ckpt_path = os.path.join(model_dir, 'model')
    if not os.path.exists(ckpt_path + '.pdparams'):
        raise ValueError(
            '`model_dir` should be a directory which include a model.pdparams file'
        )
    para_state_dict, opti_state_dict = paddle.load(ckpt_path)
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
        pred = multi_scale_flip_inference(
            model,
            im,
            im_info=im_info,
            num_classes=num_classes,
            #scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            scales=[1.0],
            flip=True,
            flip_directions=['horizontal'],
            is_slide=True,
            stride=(768, 384),
            crop_size=(1024, 512))
        # pred = multi_scale_flip_inference(model,
        #                                   im,
        #                                   im_info=im_info,
        #                                   scales=[1],
        #                                   flip=flip,
        #                                   flip_directions=['horizontal'],
        #                                   is_slide=is_slide,
        #                                   stride=stride,
        #                                   crop_size=crop_size)

        mask = label != ignore_index
        # To-DO Test Execution Time
        pred = pred.numpy()
        conf_mat.calculate(pred=pred, label=label, mask=mask)
        _, iou = conf_mat.mean_iou()

        time_iter = timer.elapsed_time()
        remain_iter = total_iters - iter - 1
        logger.info(
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

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
import math
import time

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar, TimeAverager

from ppmatting.utils import mkdir, estimate_foreground_ml, VideoReader


def save_result(alpha, path, im_path, trimap=None, fg_estimate=True):
    """
    The value of alpha is range [0, 1], shape should be [h,w]
    """
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    basename = os.path.basename(path)
    name = os.path.splitext(basename)[0]
    alpha_save_path = os.path.join(dirname, name + '_alpha.png')
    rgba_save_path = os.path.join(dirname, name + '_rgba.png')

    # save alpha matte
    if trimap is not None:
        trimap = cv2.imread(trimap, 0)
        alpha[trimap == 0] = 0
        alpha[trimap == 255] = 255
    alpha = (alpha).astype('uint8')
    cv2.imwrite(alpha_save_path, alpha)

    # save rgba
    im = cv2.imread(im_path)
    if fg_estimate:
        fg = estimate_foreground_ml(im / 255.0, alpha / 255.0) * 255
    else:
        fg = im
    fg = fg.astype('uint8')
    alpha = alpha[:, :, np.newaxis]
    rgba = np.concatenate((fg, alpha), axis=-1)
    cv2.imwrite(rgba_save_path, rgba)

    return fg


def reverse_transform(alpha, trans_info):
    """recover pred to origin shape"""
    for item in trans_info[::-1]:
        if item[0] == 'resize':
            h, w = item[1][0], item[1][1]
            alpha = F.interpolate(alpha, [h, w], mode='bilinear')
        elif item[0] == 'padding':
            h, w = item[1][0], item[1][1]
            alpha = alpha[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return alpha


def video_dataloader(video_path, transforms):
    dataset = VideoReader(video_path, transforms)
    loader = paddle.io.DataLoader(dataset)

    return loader


def predict_video(model,
                  model_path,
                  transforms,
                  video_path,
                  save_dir='output',
                  fg_estimate=True):
    """
    predict and visualize the video.

    Args:
        model (nn.Layer): Used to predict for input video.
        model_path (str): The path of pretrained model.
        transforms (transforms.Compose): Preprocess for frames of video.
        video_path (str): the video path to be predicted.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        fa
        fg_estimate (bool, optional): Whether to estimate foreground when predicting. It is invalid if the foreground is predicted by model. Default: True
    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()

    # Build DataLoader for video
    loader = video_dataloader(video_path, transforms)

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(loader), verbose=1)
    preprocess_cost_averager = TimeAverager()
    infer_cost_averager = TimeAverager()
    postprocess_cost_averager = TimeAverager()
    batch_start = time.time()
    with paddle.no_grad():
        for i, data in enumerate(loader):
            preprocess_cost_averager.record(time.time() - batch_start)

            infer_start = time.time()
            result = model(data)  # result maybe a Tensor or a dict
            if isinstance(result, paddle.Tensor):
                alpha = result
            else:
                alpha = result['alpha']
                fg = result.get('fg', None)
            print(alpha.shape, fg.shape)

            infer_cost_averager.record(time.time() - infer_start)

            # postprocess_start = time.time()
            # alpha_pred = reverse_transform(alpha_pred, data['trans_info'])
            # alpha_pred = (alpha_pred.numpy()).squeeze()
            # alpha_pred = (alpha_pred * 255).astype('uint8')

            # # get the saved name
            # if image_dir is not None:
            #     im_file = im_path.replace(image_dir, '')
            # else:
            #     im_file = os.path.basename(im_path)
            # if im_file[0] == '/' or im_file[0] == '\\':
            #     im_file = im_file[1:]

            # save_path = os.path.join(save_dir, im_file)
            # mkdir(save_path)
            # fg = save_result(
            #     alpha_pred,
            #     save_path,
            #     im_path=im_path,
            #     trimap=trimap,
            #     fg_estimate=fg_estimate)

            # postprocess_cost_averager.record(time.time() - postprocess_start)

            # preprocess_cost = preprocess_cost_averager.get_average()
            # infer_cost = infer_cost_averager.get_average()
            # postprocess_cost = postprocess_cost_averager.get_average()
            # if local_rank == 0:
            #     progbar_pred.update(i + 1,
            #                         [('preprocess_cost', preprocess_cost),
            #                          ('infer_cost cost', infer_cost),
            #                          ('postprocess_cost', postprocess_cost)])

            preprocess_cost_averager.reset()
            infer_cost_averager.reset()
            postprocess_cost_averager.reset()
            batch_start = time.time()

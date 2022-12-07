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

from ppmatting.utils import mkdir, estimate_foreground_ml, VideoReader, VideoWriter


def build_loader_writter(video_path, transforms, save_dir):
    reader = VideoReader(video_path, transforms)
    loader = paddle.io.DataLoader(reader)
    base_name = os.path.basename(video_path)
    name = os.path.splitext(base_name)[0]
    alpha_save_path = os.path.join(save_dir, name + '_alpha.avi')
    fg_save_path = os.path.join(save_dir, name + '_fg.avi')

    writer_alpha = VideoWriter(
        alpha_save_path,
        reader.fps,
        frame_size=(reader.width, reader.height),
        is_color=False)
    writer_fg = VideoWriter(
        fg_save_path,
        reader.fps,
        frame_size=(reader.width, reader.height),
        is_color=True)
    writers = {'alpha': writer_alpha, 'fg': writer_fg}

    return loader, writers


def reverse_transform(img, trans_info):
    """recover pred to origin shape"""
    for item in trans_info[::-1]:
        if item[0][0] == 'resize':
            h, w = item[1][0], item[1][1]
            img = F.interpolate(img, [h, w], mode='bilinear')
        elif item[0][0] == 'padding':
            h, w = item[1][0], item[1][1]
            img = img[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return img


def postprocess(fg, alpha, img, trans_info, writers, fg_estimate):
    """
    Postprocess for prediction results.

    Args:
        fg (Tensor): The foreground, value should be in [0, 1].
        alpha (Tensor): The alpha, value should be in [0, 1].
        img (Tensor): The original image, value should be in [0, 1].
        trans_info (list): A list of the shape transformations.
        writers (dict): A dict of VideoWriter instance.
        fg_estimate (bool): Whether to estimate foreground. It is invalid when fg is not None.

    """
    alpha = reverse_transform(alpha, trans_info)
    if fg is None:
        if fg_estimate:
            img = img.transpose((0, 2, 3, 1)).squeeze().numpy()
            alpha = alpha.squeeze().numpy()
            fg = estimate_foreground_ml(img, alpha)
        else:
            fg = img
    else:
        fg = reverse_transform(fg, trans_info)

    if len(alpha.shape) == 2:
        fg = alpha[:, :, None] * fg
    else:
        fg = alpha * fg
    writers['alpha'].write(alpha)
    writers['fg'].write(fg)


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
        fg_estimate (bool, optional): Whether to estimate foreground when predicting. It is invalid if the foreground is predicted by model. Default: True
    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()

    # Build loader and writer for video
    loader, writers = build_loader_writter(
        video_path, transforms, save_dir=save_dir)

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
                fg = None
            else:
                alpha = result['alpha']
                fg = result.get('fg', None)
            infer_cost_averager.record(time.time() - infer_start)

            postprocess_start = time.time()
            postprocess(
                fg,
                alpha,
                data['ori_img'],
                trans_info=data['trans_info'],
                writers=writers,
                fg_estimate=fg_estimate)
            postprocess_cost_averager.record(time.time() - postprocess_start)

            preprocess_cost = preprocess_cost_averager.get_average()
            infer_cost = infer_cost_averager.get_average()
            postprocess_cost = postprocess_cost_averager.get_average()
            progbar_pred.update(i + 1, [('preprocess_cost', preprocess_cost),
                                        ('infer_cost cost', infer_cost),
                                        ('postprocess_cost', postprocess_cost)])

            preprocess_cost_averager.reset()
            infer_cost_averager.reset()
            postprocess_cost_averager.reset()
            batch_start = time.time()
    if hasattr(model, 'reset'):
        model.reset()
    loader.dataset.release()
    for k, v in writers.items():
        v.release()

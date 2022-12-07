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
from collections.abc import Iterable

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar, TimeAverager

import ppmatting.transforms as T
from ppmatting.utils import mkdir, estimate_foreground_ml, VideoReader, VideoWriter


def build_loader_writter(video_path, transforms, save_dir):
    reader = VideoReader(video_path, transforms)
    loader = paddle.io.DataLoader(reader)
    base_name = os.path.basename(video_path)
    name = os.path.splitext(base_name)[0]
    save_path = os.path.join(save_dir, name + '.avi')

    writer = VideoWriter(
        save_path,
        reader.fps,
        frame_size=(reader.width, reader.height),
        is_color=True)

    return loader, writer


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


def postprocess(fg, alpha, img, bg, trans_info, writer, fg_estimate):
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
    bg = F.interpolate(bg, size=alpha.shape[-2:], mode='bilinear')
    if fg is None:
        if fg_estimate:
            img = img.transpose((0, 2, 3, 1)).squeeze().numpy()
            alpha = alpha.squeeze().numpy()
            fg = estimate_foreground_ml(img, alpha)
            bg = bg.transpose((0, 2, 3, 1)).squeeze().numpy()
        else:
            fg = img
    else:
        fg = reverse_transform(fg, trans_info)
    if len(alpha.shape) == 2:
        alpha = alpha[:, :, None]
    new_img = alpha * fg + (1 - alpha) * bg
    writer.write(new_img)


def get_bg(bg_path, shape):
    bg = paddle.zeros((1, 3, shape[0], shape[1]))
    # special color
    if bg_path == 'r':
        bg[:, 2, :, :] = 1
    elif bg_path == 'g':
        bg[:, 1, :, :] = 1
    elif bg_path == 'b':
        bg[:, 0, :, :] = 1
    elif bg_path == 'w':
        bg = bg + 1

    elif not os.path.exists(bg_path):
        raise Exception('The background path is not found: {}'.format(bg_path))
    # image
    elif bg_path.endswith(
        ('.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png')):
        bg = cv2.imread(bg_path)
        bg = bg[np.newaxis, :, :, :]
        bg = paddle.to_tensor(bg) / 255.
        bg = bg.transpose((0, 3, 1, 2))

    elif bg_path.lower().endswith(
        ('.mp4', '.avi', '.mov', '.m4v', '.dat', '.rm', '.rmvb', '.wmv', '.asf',
         '.asx', '.3gp', '.mkv', '.flv', '.vob')):
        transforms = T.Compose([T.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])
        bg = VideoReader(bg_path, transforms=transforms)
        bg = paddle.io.DataLoader(bg)
        bg = iter(bg)

    else:
        raise IOError('The background path is invalid, please check it')

    return bg


def bg_replace_video(model,
                     model_path,
                     transforms,
                     video_path,
                     bg_path='g',
                     save_dir='output',
                     fg_estimate=True):
    """
    predict and visualize the video.

    Args:
        model (nn.Layer): Used to predict for input video.
        model_path (str): The path of pretrained model.
        transforms (transforms.Compose): Preprocess for frames of video.
        video_path (str): The video path to be predicted.
        bg_path (str): The background. It can be image path or video path or a string of (r,g,b,w). Default: 'g'.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        fg_estimate (bool, optional): Whether to estimate foreground when predicting. It is invalid if the foreground is predicted by model. Default: True
    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()

    # Build loader and writer for video
    loader, writer = build_loader_writter(
        video_path, transforms, save_dir=save_dir)
    # Get bg
    bg_reader = get_bg(
        bg_path, shape=(loader.dataset.height, loader.dataset.width))

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

            # postprocess
            postprocess_start = time.time()
            if isinstance(bg_reader, Iterable):
                try:
                    bg = next(bg_reader)
                except StopIteration:
                    bg_reader = get_bg(
                        bg_path,
                        shape=(loader.dataset.height, loader.dataset.width))
                    bg = next(bg_reader)
                finally:
                    bg = bg['ori_img']
            else:
                bg = bg_reader
            postprocess(
                fg,
                alpha,
                data['ori_img'],
                bg=bg,
                trans_info=data['trans_info'],
                writer=writer,
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
    if isinstance(bg, VideoReader):
        bg_reader.release()

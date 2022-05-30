# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from ppmatting.utils import mkdir, estimate_foreground_ml


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


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


def preprocess(img, transforms, trimap=None):
    data = {}
    data['img'] = img
    if trimap is not None:
        data['trimap'] = trimap
        data['gt_fields'] = ['trimap']
    data['trans_info'] = []
    data = transforms(data)
    data['img'] = paddle.to_tensor(data['img'])
    data['img'] = data['img'].unsqueeze(0)
    if trimap is not None:
        data['trimap'] = paddle.to_tensor(data['trimap'])
        data['trimap'] = data['trimap'].unsqueeze((0, 1))

    return data


def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            trimap_list=None,
            save_dir='output',
            fg_estimate=True):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transforms.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        trimap_list (list, optional): A list of trimap of image_list. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
        trimap_lists = partition_list(
            trimap_list, nranks) if trimap_list is not None else None
    else:
        img_lists = [image_list]
        trimap_lists = [trimap_list] if trimap_list is not None else None

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    preprocess_cost_averager = TimeAverager()
    infer_cost_averager = TimeAverager()
    postprocess_cost_averager = TimeAverager()
    batch_start = time.time()
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists[local_rank]):
            preprocess_start = time.time()
            trimap = trimap_lists[local_rank][
                i] if trimap_list is not None else None
            data = preprocess(img=im_path, transforms=transforms, trimap=trimap)
            preprocess_cost_averager.record(time.time() - preprocess_start)

            infer_start = time.time()
            alpha_pred = model(data)
            infer_cost_averager.record(time.time() - infer_start)

            postprocess_start = time.time()
            alpha_pred = reverse_transform(alpha_pred, data['trans_info'])
            alpha_pred = (alpha_pred.numpy()).squeeze()
            alpha_pred = (alpha_pred * 255).astype('uint8')

            # get the saved name
            if image_dir is not None:
                im_file = im_path.replace(image_dir, '')
            else:
                im_file = os.path.basename(im_path)
            if im_file[0] == '/' or im_file[0] == '\\':
                im_file = im_file[1:]

            save_path = os.path.join(save_dir, im_file)
            mkdir(save_path)
            fg = save_result(
                alpha_pred,
                save_path,
                im_path=im_path,
                trimap=trimap,
                fg_estimate=fg_estimate)

            postprocess_cost_averager.record(time.time() - postprocess_start)

            preprocess_cost = preprocess_cost_averager.get_average()
            infer_cost = infer_cost_averager.get_average()
            postprocess_cost = postprocess_cost_averager.get_average()
            if local_rank == 0:
                progbar_pred.update(i + 1,
                                    [('preprocess_cost', preprocess_cost),
                                     ('infer_cost cost', infer_cost),
                                     ('postprocess_cost', postprocess_cost)])

            preprocess_cost_averager.reset()
            infer_cost_averager.reset()
            postprocess_cost_averager.reset()
    return alpha_pred, fg

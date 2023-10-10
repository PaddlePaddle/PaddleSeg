# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import json
import math
from unittest import result

import cv2
import numpy as np
import paddle
from PIL import Image

from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar, visualize, metrics


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def preprocess(im_path, transforms):
    data = {}
    data['img'] = im_path
    data = transforms(data)
    data['img'] = data['img'][np.newaxis, ...]
    data['img'] = paddle.to_tensor(data['img'])
    return data


def analyse(model,
            model_path,
            transforms,
            val_dataset,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None,
            custom_color=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        val_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    file_list = val_dataset.file_list
    dataset_root = val_dataset.dataset_root
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        paddle.distributed.init_parallel_env()
        file_list = partition_list(file_list, nranks)
    else:
        file_list = [file_list]

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
    results = {}

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(file_list[0]), verbose=1)
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        for i, (im_path, label_path) in enumerate(file_list[local_rank]):
            data = preprocess(im_path, transforms)

            if aug_pred:
                pred, _ = infer.aug_inference(
                    model,
                    data['img'],
                    trans_info=data['trans_info'],
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred, _ = infer.inference(
                    model,
                    data['img'],
                    trans_info=data['trans_info'],
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            pred = paddle.squeeze(pred)

            # calculate miou for the image
            label = paddle.to_tensor(
                np.asarray(Image.open(label_path)), dtype=pred.dtype)
            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred, label, val_dataset.num_classes)
            class_iou_per_img, miou_per_img = metrics.mean_iou(
                intersect_area, pred_area, label_area)
            results[im_path] = {
                'miou': miou_per_img,
                'class_iou': list(class_iou_per_img),
                'label_path': label_path
            }

            pred = pred.numpy().astype('uint8')
            # get the saved name
            if dataset_root is not None:
                im_file = im_path.replace(dataset_root, '')
            else:
                im_file = os.path.basename(im_path)
            if im_file[0] == '/' or im_file[0] == '\\':
                im_file = im_file[1:]

            # save added image
            added_image = utils.visualize.visualize(
                im_path, pred, color_map, weight=0.6)
            added_image_path = os.path.join(added_saved_dir, im_file)
            mkdir(added_image_path)
            cv2.imwrite(added_image_path, added_image)
            results[im_path]['added_path'] = added_image_path

            # save pseudo color prediction
            pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
            pred_saved_path = os.path.join(
                pred_saved_dir, os.path.splitext(im_file)[0] + ".png")
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)
            results[im_path]['prediction_path'] = pred_saved_path

            progbar_pred.update(i + 1)
    if nranks > 1:
        results_list = []
        paddle.distributed.all_gather_object(results_list, results)
        if local_rank == 0:
            results = {}
            for d in results_list:
                results.update(d)
    if local_rank == 0:
        with open(os.path.join(save_dir, 'analysis_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

    logger.info("Samples analysis finished, the results are save in {}.".format(
        save_dir))

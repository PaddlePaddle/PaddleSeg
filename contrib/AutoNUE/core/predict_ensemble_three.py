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

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F

from paddleseg import utils
import core.infer_ensemble_three as infer_ensemble
import core.infer_crop as infer_crop
from paddleseg.utils import logger, progbar


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def predictEnsembleThree(model,
                         model_1,
                         model_crop,
                         model_path,
                         model_path_1,
                         model_path_crop,
                         transforms,
                         transforms_crop,
                         image_list,
                         image_dir=None,
                         save_dir='output',
                         aug_pred=False,
                         scales=1.0,
                         flip_horizontal=True,
                         flip_vertical=False,
                         is_slide=False,
                         stride=None,
                         crop_size=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
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

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    utils.utils.load_entire_model(model_1, model_path_1)
    model_1.eval()
    utils.utils.load_entire_model(model_crop, model_path_crop)
    model_crop.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists[local_rank]):
            im_origin = cv2.imread(im_path)
            ori_shape = im_origin.shape[:2]
            im, _ = transforms(im_origin)
            im = im[np.newaxis, ...]
            im = paddle.to_tensor(im)

            ims, _ = transforms_crop(im_origin)
            im1 = ims[:, 540:540 + 720, 320:320 + 1280]
            im2 = ims[:, 540:540 + 720, 960:960 + 1280]
            im3 = ims[:, 540:540 + 720, 1600:1600 + 1280]
            im1 = im1[np.newaxis, ...]
            im1 = paddle.to_tensor(im1)
            im2 = im2[np.newaxis, ...]
            im2 = paddle.to_tensor(im2)
            im3 = im3[np.newaxis, ...]
            im3 = paddle.to_tensor(im3)
            ims_ = [im1, im2, im3]

            if aug_pred:
                pred = infer_ensemble.aug_inference(
                    model,
                    model_1,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred = infer_ensemble.inference(
                    model,
                    model_1,
                    im,
                    ori_shape=ori_shape,
                    transforms=transforms.transforms,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            preds = []
            for ii in range(3):
                im_ = ims_[ii]
                if aug_pred:
                    pred_crop = infer_crop.aug_inference(
                        model,
                        im_,
                        ori_shape=ori_shape,
                        transforms=transforms.transforms,
                        scales=scales,
                        flip_horizontal=flip_horizontal,
                        flip_vertical=flip_vertical,
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
                else:
                    pred_crop = infer_crop.inference(
                        model,
                        im_,
                        ori_shape=ori_shape,
                        transforms=transforms.transforms,
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
                preds.append(pred_crop)

            left_ensem = (
                preds[0][:, :, :, 640:1280] + preds[1][:, :, :, 0:640]) / 2
            right_ensem = (
                preds[1][:, :, :, 640:1280] + preds[2][:, :, :, 0:640]) / 2
            pred_ensem = paddle.concat(
                [
                    preds[0][:, :, :, 0:640], left_ensem, right_ensem,
                    preds[2][:, :, :, 640:1280]
                ],
                axis=3)
            logit = F.interpolate(pred_ensem, (432, 768), mode='bilinear')

            pred_logit = pred.clone()
            pred_logit[:, :, 324:756, 576:1344] = logit
            pred = pred + pred_logit
            pred = F.interpolate(pred, ori_shape, mode='bilinear')
            pred = paddle.argmax(pred, axis=1, keepdim=True, dtype='int32')
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')

            # get the saved name
            if image_dir is not None:
                im_file = im_path.replace(image_dir, '')
            else:
                im_file = os.path.basename(im_path)
            if im_file[0] == '/':
                im_file = im_file[1:]

            # save added image
            added_image = utils.visualize.visualize(im_path, pred, weight=0.6)
            added_image_path = os.path.join(added_saved_dir, im_file)
            mkdir(added_image_path)
            cv2.imwrite(added_image_path, added_image)

            # save pseudo color prediction
            pred_mask = utils.visualize.get_pseudo_color_map(pred)
            pred_saved_path = os.path.join(pred_saved_dir,
                                           im_file.rsplit(".")[0] + ".png")
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)

            # pred_im = utils.visualize(im_path, pred, weight=0.0)
            # pred_saved_path = os.path.join(pred_saved_dir, im_file)
            # mkdir(pred_saved_path)
            # cv2.imwrite(pred_saved_path, pred_im)

            progbar_pred.update(i + 1)

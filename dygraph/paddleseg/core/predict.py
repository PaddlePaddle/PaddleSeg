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

from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def predict(model,
            model_path,
            transforms,
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
        image_list (list): A list of images to be predicted.
        image_dir (str): The directory of the images to be predicted. Default: None.
        save_dir (str): The directory to save the visualized results. Default: 'output'.

    """
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    model.eval()

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(image_list), verbose=1)
    for i, im_path in enumerate(image_list):
        im = cv2.imread(im_path)
        ori_shape = im.shape[:2]
        im, _ = transforms(im)
        im = im[np.newaxis, ...]
        im = paddle.to_tensor(im)

        if aug_pred:
            pred = infer.aug_inference(
                model,
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
            pred = infer.inference(
                model,
                im,
                ori_shape=ori_shape,
                transforms=transforms.transforms,
                is_slide=is_slide,
                stride=stride,
                crop_size=crop_size)
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
        added_image = utils.visualize(im_path, pred, weight=0.6)
        added_image_path = os.path.join(added_saved_dir, im_file)
        mkdir(added_image_path)
        cv2.imwrite(added_image_path, added_image)

        # save pseudo color prediction
        pred_im = utils.visualize(im_path, pred, weight=0.0)
        pred_saved_path = os.path.join(pred_saved_dir, im_file)
        mkdir(pred_saved_path)
        cv2.imwrite(pred_saved_path, pred_im)

        progbar_pred.update(i + 1)

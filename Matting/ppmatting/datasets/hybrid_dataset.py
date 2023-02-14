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
import math

import cv2
import numpy as np
import random
import paddle
from paddleseg.cvlibs import manager
from ppmatting.datasets import MattingDataset

import ppmatting.transforms as T



@manager.DATASETS.add_component
class HybridDataset(MattingDataset):
    """
    Pass in a dataset that confirms to the format.
    matting_dataset/
        |--bg/
        |
        |--dataset1/
        |  |--train/
        |       |--fg/
        |       |--alpha/
        |       |--trimap/ (if existing) 
        |
        |  |--dataset2/
        |       |--fg/
        |       |--alpha/
        |       |--trimap/ (if existing)
        | ....

    Mix training using two or more of the above datasets.
    Prepare a hybrid dataset, use `create_hybrid_list.py` to generate `train.txt`.
    Please make sure all datasets are placed in the same folder.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def composite(self, fg, alpha, ori_bg):
        if self.if_rssn:
            if np.random.rand() < 0.5:
                fg = cv2.fastNlMeansDenoisingColored(fg, None, 3, 3, 7, 21)
                ori_bg = cv2.fastNlMeansDenoisingColored(ori_bg, None, 3, 3, 7,
                                                         21)
            if np.random.rand() < 0.5:
                radius = np.random.choice([19, 29, 39, 49, 59])
                ori_bg = cv2.GaussianBlur(ori_bg, (radius, radius), 0, 0)
        fg_h, fg_w = fg.shape[:2]
        ori_bg_h, ori_bg_w = ori_bg.shape[:2]

        wratio = fg_w / ori_bg_w
        hratio = fg_h / ori_bg_h
        ratio = wratio if wratio > hratio else hratio

        # Resize ori_bg if it is smaller than fg.
        if ratio > 1:
            resize_h = math.ceil(ori_bg_h * ratio)
            resize_w = math.ceil(ori_bg_w * ratio)
            bg = cv2.resize(
                ori_bg, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        else:
            bg = ori_bg

        bg = bg[0:fg_h, 0:fg_w, :]

        if random.random() < 0.5:
            rand_kernel = random.choice([20, 30, 40, 50, 60])
            bg = cv2.blur(bg, (rand_kernel, rand_kernel))

        alpha = alpha / 255.
        alpha = np.expand_dims(alpha, axis=2)
        image = alpha * fg + (1 - alpha) * bg
        image = image.astype(np.uint8)

        if random.random() < 0.5:
            image, fg, bg = self.add_gaussian_noise(image, fg, bg)

        return image, fg, bg
    
    def add_gaussian_noise(self, image, fg, bg):
        r, c, ch = image.shape
        mean = 0
        sigma = 10

        gauss = np.random.normal(mean, sigma, (r, c, ch))
        gauss = gauss.reshape(r, c, ch)

        noise_img = np.uint8(image + gauss)
        noise_fg = np.uint8(fg + gauss)
        noise_bg = np.uint8(bg + gauss)

        return noise_img, noise_fg, noise_bg

    @staticmethod
    def gen_trimap(alpha, mode='train', eval_kernel=7):
        if mode == 'train':
            k_size = 30
            iterations = 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (k_size, k_size))
            dilated = cv2.dilate(alpha, kernel, iterations=iterations)
            eroded = cv2.erode(alpha, kernel, iterations=iterations)
            trimap = np.zeros(alpha.shape)
            trimap.fill(128)
            trimap[eroded > 254.5] = 255
            trimap[dilated < 0.5] = 0
        else:
            k_size = eval_kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (k_size, k_size))
            dilated = cv2.dilate(alpha, kernel)
            trimap = np.zeros(alpha.shape)
            trimap.fill(128)
            trimap[alpha >= 250] = 255
            trimap[dilated <= 5] = 0

        return trimap


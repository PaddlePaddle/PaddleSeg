# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import print_function
import cv2
import numpy as np
from utils.config import cfg
from models.model_builder import ModelPhase
from pdseg.data_aug import get_random_scale, randomly_scale_image_and_label, random_rotation, \
    rand_scale_aspect, hsv_color_jitter, rand_crop


def resize(img, grt=None, grt_instance=None, mode=ModelPhase.TRAIN):
    """
    改变图像及标签图像尺寸
    AUG.AUG_METHOD为unpadding，所有模式均直接resize到AUG.FIX_RESIZE_SIZE的尺寸
    AUG.AUG_METHOD为stepscaling, 按比例resize，训练时比例范围AUG.MIN_SCALE_FACTOR到AUG.MAX_SCALE_FACTOR,间隔为AUG.SCALE_STEP_SIZE，其他模式返回原图
    AUG.AUG_METHOD为rangescaling，长边对齐，短边按比例变化，训练时长边对齐范围AUG.MIN_RESIZE_VALUE到AUG.MAX_RESIZE_VALUE，其他模式长边对齐AUG.INF_RESIZE_VALUE

    Args：
        img(numpy.ndarray): 输入图像
        grt(numpy.ndarray): 标签图像，默认为None
        mode(string): 模式, 默认训练模式，即ModelPhase.TRAIN

    Returns：
        resize后的图像和标签图

    """

    if cfg.AUG.AUG_METHOD == 'unpadding':
        target_size = cfg.AUG.FIX_RESIZE_SIZE
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        if grt is not None:
            grt = cv2.resize(grt, target_size, interpolation=cv2.INTER_NEAREST)
        if grt_instance is not None:
            grt_instance = cv2.resize(
                grt_instance, target_size, interpolation=cv2.INTER_NEAREST)
    elif cfg.AUG.AUG_METHOD == 'stepscaling':
        if mode == ModelPhase.TRAIN:
            min_scale_factor = cfg.AUG.MIN_SCALE_FACTOR
            max_scale_factor = cfg.AUG.MAX_SCALE_FACTOR
            step_size = cfg.AUG.SCALE_STEP_SIZE
            scale_factor = get_random_scale(min_scale_factor, max_scale_factor,
                                            step_size)
            img, grt = randomly_scale_image_and_label(
                img, grt, scale=scale_factor)
    elif cfg.AUG.AUG_METHOD == 'rangescaling':
        min_resize_value = cfg.AUG.MIN_RESIZE_VALUE
        max_resize_value = cfg.AUG.MAX_RESIZE_VALUE
        if mode == ModelPhase.TRAIN:
            if min_resize_value == max_resize_value:
                random_size = min_resize_value
            else:
                random_size = int(
                    np.random.uniform(min_resize_value, max_resize_value) + 0.5)
        else:
            random_size = cfg.AUG.INF_RESIZE_VALUE

        value = max(img.shape[0], img.shape[1])
        scale = float(random_size) / float(value)
        img = cv2.resize(
            img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if grt is not None:
            grt = cv2.resize(
                grt, (0, 0),
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST)
    else:
        raise Exception("Unexpect data augmention method: {}".format(
            cfg.AUG.AUG_METHOD))

    return img, grt, grt_instance

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
import sys
import os
import math
import random
import functools
import io
import time
import codecs

import numpy as np
import paddle
import paddle.fluid as fluid
import cv2
from PIL import Image

import data_aug as aug
from utils.config import cfg
from data_utils import GeneratorEnqueuer
from models.model_builder import ModelPhase
import copy


def pil_imread(file_path):
    """read pseudo-color label"""
    im = Image.open(file_path)
    return np.asarray(im)


def cv2_imread(file_path, flag=cv2.IMREAD_COLOR):
    # resolve cv2.imread open Chinese file path issues on Windows Platform.
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)


class SegDataset(object):
    def __init__(self,
                 file_list,
                 data_dir,
                 shuffle=False,
                 mode=ModelPhase.TRAIN):
        self.mode = mode
        self.shuffle = shuffle
        self.data_dir = data_dir

        self.shuffle_seed = 0
        # NOTE: Please ensure file list was save in UTF-8 coding format
        with codecs.open(file_list, 'r', 'utf-8') as flist:
            self.lines = [line.strip() for line in flist]
            self.all_lines = copy.deepcopy(self.lines)
            if shuffle and cfg.NUM_TRAINERS > 1:
                np.random.RandomState(self.shuffle_seed).shuffle(self.all_lines)
            elif shuffle:
                np.random.shuffle(self.lines)

    def generator(self):
        if self.shuffle and cfg.NUM_TRAINERS > 1:
            np.random.RandomState(self.shuffle_seed).shuffle(self.all_lines)
            num_lines = len(self.all_lines) // cfg.NUM_TRAINERS
            self.lines = self.all_lines[num_lines * cfg.TRAINER_ID:num_lines *
                                        (cfg.TRAINER_ID + 1)]
            self.shuffle_seed += 1
        elif self.shuffle:
            np.random.shuffle(self.lines)

        for line in self.lines:
            yield self.process_image(line, self.data_dir, self.mode)

    def sharding_generator(self, pid=0, num_processes=1):
        """
        Use line id as shard key for multiprocess io
        It's a normal generator if pid=0, num_processes=1
        """
        for index, line in enumerate(self.lines):
            # Use index and pid to shard file list
            if index % num_processes == pid:
                yield self.process_image(line, self.data_dir, self.mode)

    def batch_reader(self, batch_size):
        br = self.batch(self.reader, batch_size)
        for batch in br:
            yield batch[0], batch[1], batch[2]

    def multiprocess_generator(self, max_queue_size=32, num_processes=8):
        # Re-shuffle file list
        if self.shuffle and cfg.NUM_TRAINERS > 1:
            np.random.RandomState(self.shuffle_seed).shuffle(self.all_lines)
            num_lines = len(self.all_lines) // cfg.NUM_TRAINERS
            self.lines = self.all_lines[num_lines * cfg.TRAINER_ID:num_lines *
                                        (cfg.TRAINER_ID + 1)]
            self.shuffle_seed += 1
        elif self.shuffle:
            np.random.shuffle(self.lines)

        # Create multiple sharding generators according to num_processes for multiple processes
        generators = []
        for pid in range(num_processes):
            generators.append(self.sharding_generator(pid, num_processes))

        try:
            enqueuer = GeneratorEnqueuer(generators)
            enqueuer.start(max_queue_size=max_queue_size, workers=num_processes)
            while True:
                generator_out = None
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        generator_out = enqueuer.queue.get(timeout=5)
                        break
                    else:
                        time.sleep(0.01)
                if generator_out is None:
                    break
                yield generator_out
        finally:
            if enqueuer is not None:
                enqueuer.stop()

    def batch(self, reader, batch_size, is_test=False, drop_last=False):
        def batch_reader(is_test=False, drop_last=drop_last):
            if is_test:
                imgs, grts, img_names, valid_shapes, org_shapes = [], [], [], [], []
                for img, grt, img_name, valid_shape, org_shape in reader():
                    imgs.append(img)
                    grts.append(grt)
                    img_names.append(img_name)
                    valid_shapes.append(valid_shape)
                    org_shapes.append(org_shape)
                    if len(imgs) == batch_size:
                        yield np.array(imgs), np.array(
                            grts), img_names, np.array(valid_shapes), np.array(
                                org_shapes)
                        imgs, grts, img_names, valid_shapes, org_shapes = [], [], [], [], []

                if not drop_last and len(imgs) > 0:
                    yield np.array(imgs), np.array(grts), img_names, np.array(
                        valid_shapes), np.array(org_shapes)
            else:
                imgs, labs, ignore = [], [], []
                bs = 0
                for img, lab, ig in reader():
                    imgs.append(img)
                    labs.append(lab)
                    ignore.append(ig)
                    bs += 1
                    if bs == batch_size:
                        yield np.array(imgs), np.array(labs), np.array(ignore)
                        bs = 0
                        imgs, labs, ignore = [], [], []

                if not drop_last and bs > 0:
                    yield np.array(imgs), np.array(labs), np.array(ignore)

        return batch_reader(is_test, drop_last)

    def load_image(self, line, src_dir, mode=ModelPhase.TRAIN):
        # original image cv2.imread flag setting
        cv2_imread_flag = cv2.IMREAD_COLOR
        if cfg.DATASET.IMAGE_TYPE == "rgba":
            # If use RBGA 4 channel ImageType, use IMREAD_UNCHANGED flags to
            # reserver alpha channel
            cv2_imread_flag = cv2.IMREAD_UNCHANGED

        parts = line.strip().split(cfg.DATASET.SEPARATOR)
        if len(parts) != 2:
            if mode == ModelPhase.TRAIN or mode == ModelPhase.EVAL:
                raise Exception("File list format incorrect! It should be"
                                " image_name{}label_name\\n".format(
                                    cfg.DATASET.SEPARATOR))
            img_name, grt_name = parts[0], None
        else:
            img_name, grt_name = parts[0], parts[1]

        img_path = os.path.join(src_dir, img_name)
        img = cv2_imread(img_path, cv2_imread_flag)

        if grt_name is not None:
            grt_path = os.path.join(src_dir, grt_name)
            grt = pil_imread(grt_path)
        else:
            grt = None

        if img is None:
            raise Exception(
                "Empty image, source image path: {}".format(img_path))

        img_height = img.shape[0]
        img_width = img.shape[1]

        if grt is not None:
            grt_height = grt.shape[0]
            grt_width = grt.shape[1]

            if img_height != grt_height or img_width != grt_width:
                raise Exception(
                    "Source img and label img must has the same size.")
        else:
            if mode == ModelPhase.TRAIN or mode == ModelPhase.EVAL:
                raise Exception(
                    "No laber image path for image '{}' when training or evaluating. "
                    .format(img_path))

        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img_channels = img.shape[2]
        if img_channels < 3:
            raise Exception("PaddleSeg only supports gray, rgb or rgba image")
        if img_channels != cfg.DATASET.DATA_DIM:
            raise Exception(
                "Input image channel({}) is not match cfg.DATASET.DATA_DIM({}), img_name={}"
                .format(img_channels, cfg.DATASET.DATADIM, img_name))
        if img_channels != len(cfg.MEAN):
            raise Exception(
                "Image name {}, image channels {} do not equal the length of cfg.MEAN {}."
                .format(img_name, img_channels, len(cfg.MEAN)))
        if img_channels != len(cfg.STD):
            raise Exception(
                "Image name {}, image channels {} do not equal the length of cfg.STD {}."
                .format(img_name, img_channels, len(cfg.STD)))

        return img, grt, img_name, grt_name

    def normalize_image(self, img):
        """ 像素归一化后减均值除方差 """
        img = img.transpose((2, 0, 1)).astype('float32') / 255.0
        img_mean = np.array(cfg.MEAN).reshape((len(cfg.MEAN), 1, 1))
        img_std = np.array(cfg.STD).reshape((len(cfg.STD), 1, 1))
        img -= img_mean
        img /= img_std

        return img

    def process_image(self, line, data_dir, mode):
        """ process_image """
        img, grt, img_name, grt_name = self.load_image(
            line, data_dir, mode=mode)
        if mode == ModelPhase.TRAIN:
            img, grt = aug.resize(img, grt, mode)
            if cfg.AUG.RICH_CROP.ENABLE:
                if cfg.AUG.RICH_CROP.BLUR:
                    if cfg.AUG.RICH_CROP.BLUR_RATIO <= 0:
                        n = 0
                    elif cfg.AUG.RICH_CROP.BLUR_RATIO >= 1:
                        n = 1
                    else:
                        n = int(1.0 / cfg.AUG.RICH_CROP.BLUR_RATIO)
                    if n > 0:
                        if np.random.randint(0, n) == 0:
                            radius = np.random.randint(3, 10)
                            if radius % 2 != 1:
                                radius = radius + 1
                            if radius > 9:
                                radius = 9
                            img = cv2.GaussianBlur(img, (radius, radius), 0, 0)

                img, grt = aug.random_rotation(
                    img,
                    grt,
                    rich_crop_max_rotation=cfg.AUG.RICH_CROP.MAX_ROTATION,
                    mean_value=cfg.DATASET.PADDING_VALUE)

                img, grt = aug.rand_scale_aspect(
                    img,
                    grt,
                    rich_crop_min_scale=cfg.AUG.RICH_CROP.MIN_AREA_RATIO,
                    rich_crop_aspect_ratio=cfg.AUG.RICH_CROP.ASPECT_RATIO)
                img = aug.hsv_color_jitter(
                    img,
                    brightness_jitter_ratio=cfg.AUG.RICH_CROP.
                    BRIGHTNESS_JITTER_RATIO,
                    saturation_jitter_ratio=cfg.AUG.RICH_CROP.
                    SATURATION_JITTER_RATIO,
                    contrast_jitter_ratio=cfg.AUG.RICH_CROP.
                    CONTRAST_JITTER_RATIO)

            if cfg.AUG.FLIP:
                if cfg.AUG.FLIP_RATIO <= 0:
                    n = 0
                elif cfg.AUG.FLIP_RATIO >= 1:
                    n = 1
                else:
                    n = int(1.0 / cfg.AUG.FLIP_RATIO)
                if n > 0:
                    if np.random.randint(0, n) == 0:
                        img = img[::-1, :, :]
                        grt = grt[::-1, :]

            if cfg.AUG.MIRROR:
                if np.random.randint(0, 2) == 1:
                    img = img[:, ::-1, :]
                    grt = grt[:, ::-1]

            img, grt = aug.rand_crop(img, grt, mode=mode)
        elif ModelPhase.is_eval(mode):
            img, grt = aug.resize(img, grt, mode=mode)
            img, grt = aug.rand_crop(img, grt, mode=mode)
        elif ModelPhase.is_visual(mode):
            org_shape = [img.shape[0], img.shape[1]]
            img, grt = aug.resize(img, grt, mode=mode)
            valid_shape = [img.shape[0], img.shape[1]]
            img, grt = aug.rand_crop(img, grt, mode=mode)
        else:
            raise ValueError("Dataset mode={} Error!".format(mode))

        # Normalize image
        if cfg.AUG.TO_RGB:
            img = img[..., ::-1]
        img = self.normalize_image(img)

        if ModelPhase.is_train(mode) or ModelPhase.is_eval(mode):
            grt = np.expand_dims(np.array(grt).astype('int32'), axis=0)
            ignore = (grt != cfg.DATASET.IGNORE_INDEX).astype('int32')

        if ModelPhase.is_train(mode):
            return (img, grt, ignore)
        elif ModelPhase.is_eval(mode):
            return (img, grt, ignore)
        elif ModelPhase.is_visual(mode):
            return (img, grt, img_name, valid_shape, org_shape)

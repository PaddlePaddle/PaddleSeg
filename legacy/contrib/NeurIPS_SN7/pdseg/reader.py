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
        if cfg.DATASET.INPUT_IMAGE_NUM == 1:
            for batch in br:
                yield batch[0], batch[1], batch[2]
        else:
            for batch in br:
                yield batch[0], batch[1], batch[2], batch[3]

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
            if cfg.DATASET.INPUT_IMAGE_NUM == 1:
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
                                grts), img_names, np.array(
                                    valid_shapes), np.array(org_shapes)
                            imgs, grts, img_names, valid_shapes, org_shapes = [], [], [], [], []

                    if not drop_last and len(imgs) > 0:
                        yield np.array(imgs), np.array(
                            grts), img_names, np.array(valid_shapes), np.array(
                                org_shapes)
                else:
                    imgs, labs, ignore = [], [], []
                    bs = 0
                    for img, lab, ig in reader():
                        imgs.append(img)
                        labs.append(lab)
                        ignore.append(ig)
                        bs += 1
                        if bs == batch_size:
                            yield np.array(imgs), np.array(labs), np.array(
                                ignore)
                            bs = 0
                            imgs, labs, ignore = [], [], []

                    if not drop_last and bs > 0:
                        yield np.array(imgs), np.array(labs), np.array(ignore)
            else:
                if is_test:
                    img1s, img2s, grts, img1_names, img2_names, valid_shapes, org_shapes = [], [], [], [], [], [], []
                    for img1, img2, grt, img1_name, img2_name, valid_shape, org_shape in reader(
                    ):
                        img1s.append(img1)
                        img2s.append(img2)
                        grts.append(grt)
                        img1_names.append(img1_name)
                        img2_names.append(img2_name)
                        valid_shapes.append(valid_shape)
                        org_shapes.append(org_shape)
                        if len(img1s) == batch_size:
                            yield np.array(img1s), np.array(img2s), np.array(grts), \
                                  img1_names, img2_names, np.array(valid_shapes), np.array(org_shapes)
                            img1s, img2s, grts, img1_names, img2_names, valid_shapes, org_shapes = [], [], [], [], [], [], []

                    if not drop_last and len(img1s) > 0:
                        yield np.array(img1s), np.array(img2s), np.array(grts), \
                              img1_names, img2_names, np.array(valid_shapes), np.array(org_shapes)
                else:
                    img1s, img2s, labs, ignore = [], [], [], []
                    bs = 0
                    for img1, img2, lab, ig in reader():
                        img1s.append(img1)
                        img2s.append(img2)
                        labs.append(lab)
                        ignore.append(ig)
                        bs += 1
                        if bs == batch_size:
                            yield np.array(img1s), np.array(img2s), np.array(
                                labs), np.array(ignore)
                            bs = 0
                            img1s, img2s, labs, ignore = [], [], [], []

                    if not drop_last and bs > 0:
                        yield np.array(img1s), np.array(img2s), np.array(
                            labs), np.array(ignore)

        return batch_reader(is_test, drop_last)

    def load_image(self, line, src_dir, mode=ModelPhase.TRAIN):
        # original image cv2.imread flag setting
        cv2_imread_flag = cv2.IMREAD_COLOR
        if cfg.DATASET.IMAGE_TYPE == "rgba":
            # If use RBGA 4 channel ImageType, use IMREAD_UNCHANGED flags to
            # reserver alpha channel
            cv2_imread_flag = cv2.IMREAD_UNCHANGED

        parts = line.strip().split(cfg.DATASET.SEPARATOR)

        if len(parts) == 1:
            img1_name, img2_name, grt1_name, grt2_name = parts[
                0], None, None, None
        elif len(parts) == 2:
            if cfg.DATASET.INPUT_IMAGE_NUM == 1:
                img1_name, img2_name, grt1_name, grt2_name = parts[
                    0], None, parts[1], None
            else:
                img1_name, img2_name, grt1_name, grt2_name = parts[0], parts[
                    1], None, None
        elif len(parts) == 3:
            img1_name, img2_name, grt1_name, grt2_name = parts[0], parts[
                1], parts[2], None
        elif len(parts) == 4:
            img1_name, img2_name, grt1_name, grt2_name = parts[0], parts[
                1], parts[2], parts[3]
        else:
            raise Exception("File list format incorrect! It should be"
                            " image_name{}label_name\\n".format(
                                cfg.DATASET.SEPARATOR))

        # read input image 1
        img1_path = os.path.join(src_dir, img1_name)
        img1 = cv2_imread(img1_path, cv2_imread_flag)
        if img1 is None:
            raise Exception("Empty image, src_dir: {}, img: {}".format(
                src_dir, img1_path))
        if len(img1.shape) < 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

        # read input image 2
        if img2_name is not None:
            img2_path = os.path.join(src_dir, img2_name)
            img2 = cv2_imread(img2_path, cv2_imread_flag)
            if img2 is None:
                raise Exception("Empty image, src_dir: {}, img: {}".format(
                    src_dir, img2_path))
            if len(img2.shape) < 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            if img1.shape != img2.shape:
                raise Exception(
                    "source img1 and source img2 must has the same size")
        else:
            img2 = None

        # read input label image
        if grt1_name is not None:
            grt1_path = os.path.join(src_dir, grt1_name)
            grt1 = pil_imread(grt1_path)
            if grt1 is None:
                raise Exception("Empty image, src_dir: {}, label: {}".format(
                    src_dir, grt1_path))
            grt1_height = grt1.shape[0]
            grt1_width = grt1.shape[1]
            img1_height = img1.shape[0]
            img1_width = img1.shape[1]
            if img1_height != grt1_height or img1_width != grt1_width:
                raise Exception(
                    "source img and label img must has the same size")
        else:
            grt1 = None

        if grt2_name is not None:
            grt2_path = os.path.join(src_dir, grt2_name)
            grt2 = pil_imread(grt2_path)
            if grt2 is None:
                raise Exception("Empty image, src_dir: {}, label: {}".format(
                    src_dir, grt2_path))
            grt2_height = grt2.shape[0]
            grt2_width = grt2.shape[1]
            img2_height = img2.shape[0]
            img2_width = img2.shape[1]
            if img2_height != grt2_height or img2_width != grt2_width:
                raise Exception(
                    "source img and label img must has the same size")
        else:
            grt2 = None

        img_channels = img1.shape[2]
        if img_channels < 3:
            raise Exception("PaddleSeg only supports gray, rgb or rgba image")
        if img_channels != cfg.DATASET.DATA_DIM:
            raise Exception(
                "Input image channel({}) is not match cfg.DATASET.DATA_DIM({}), img_name={}"
                .format(img_channels, cfg.DATASET.DATADIM, img1_name))
        if img_channels != len(cfg.MEAN):
            raise Exception(
                "img name {}, img chns {} mean size {}, size unequal".format(
                    img1_name, img_channels, len(cfg.MEAN)))
        if img_channels != len(cfg.STD):
            raise Exception(
                "img name {}, img chns {} std size {}, size unequal".format(
                    img1_name, img_channels, len(cfg.STD)))

        return img1, img2, grt1, grt2, img1_name, img2_name, grt1_name, grt2_name

    def normalize_image(self, img):
        """ 像素归一化后减均值除方差 """
        img = img.transpose((2, 0, 1)).astype('float32') / 255.0
        img_mean = np.array(cfg.MEAN).reshape((len(cfg.MEAN), 1, 1))
        img_std = np.array(cfg.STD).reshape((len(cfg.STD), 1, 1))
        if img.shape[0] > 3:
            tile_times = img.shape[0] // 3
            img_mean = np.tile(img_mean, (tile_times, 1, 1))
            img_std = np.tile(img_std, (tile_times, 1, 1))
        img -= img_mean
        img /= img_std
        return img

    def test_aug(self, img1):
        ret = img1
        for ops in cfg.TEST.TEST_AUG_FLIP_OPS:
            if ops[0] == 'h':
                ret = np.concatenate((ret, img1[::-1, :, :]), axis=2)
            elif ops[0] == 'v':
                ret = np.concatenate((ret, img1[:, ::-1, :]), axis=2)
            elif ops[0] == 'm':
                ret = np.concatenate((ret, np.transpose(img1, (1, 0, 2))),
                                     axis=2)
            else:
                ret = np.concatenate(
                    (ret, np.transpose(np.rot90(img1, k=2), (1, 0, 2))), axis=2)

        for angle in cfg.TEST.TEST_AUG_ROTATE_OPS:
            ret = np.concatenate((ret, np.rot90(img1, k=angle // 90)), axis=2)

        return ret

    def process_image(self, line, data_dir, mode):
        """ process_image """
        img1, img2, grt1, grt2, img1_name, img2_name, grt1_name, grt2_name = self.load_image(
            line, data_dir, mode=mode)
        grt1 = grt1 + 1 if grt1 is not None else None
        if mode == ModelPhase.TRAIN:
            img1, img2, grt1, grt2 = aug.resize(img1, img2, grt1, grt2, mode)
            img1, img2, grt1, grt2 = aug.rand_crop(
                img1, img2, grt1, grt2, mode=mode)
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
                            img1 = cv2.GaussianBlur(img1, (radius, radius), 0,
                                                    0)
                            if img2 is not None:
                                img2 = cv2.GaussianBlur(img2, (radius, radius),
                                                        0, 0)

                img1, img2, grt1, grt2 = aug.random_rotation(
                    img1,
                    img2,
                    grt1,
                    grt2,
                    rich_crop_max_rotation=cfg.AUG.RICH_CROP.MAX_ROTATION,
                    mean_value=cfg.DATASET.PADDING_VALUE)

                img1, img2, grt1, grt2 = aug.rand_scale_aspect(
                    img1,
                    img2,
                    grt1,
                    grt2,
                    rich_crop_min_scale=cfg.AUG.RICH_CROP.MIN_AREA_RATIO,
                    rich_crop_aspect_ratio=cfg.AUG.RICH_CROP.ASPECT_RATIO)

                img1, img2 = aug.hsv_color_jitter(
                    img1,
                    img2,
                    brightness_jitter_ratio=cfg.AUG.RICH_CROP.
                    BRIGHTNESS_JITTER_RATIO,
                    saturation_jitter_ratio=cfg.AUG.RICH_CROP.
                    SATURATION_JITTER_RATIO,
                    contrast_jitter_ratio=cfg.AUG.RICH_CROP.
                    CONTRAST_JITTER_RATIO)

            if cfg.AUG.RANDOM_ROTATION90:
                rot_k = np.random.randint(0, 4)
                img1 = np.rot90(img1, k=rot_k)
                img2 = np.rot90(img2, k=rot_k) if img2 is not None else None
                grt1 = np.rot90(grt1, k=rot_k)
                grt2 = np.rot90(grt2, k=rot_k) if grt2 is not None else None

            if cfg.AUG.FLIP:
                if cfg.AUG.FLIP_RATIO <= 0:
                    n = 0
                elif cfg.AUG.FLIP_RATIO >= 1:
                    n = 1
                else:
                    n = int(1.0 / cfg.AUG.FLIP_RATIO)
                if n > 0:
                    if np.random.randint(0, n) == 0:
                        img1 = img1[::-1, :, :]
                        img2 = img2[::-1, :, :] if img2 is not None else None
                        grt1 = grt1[::-1, :]
                        grt2 = grt2[::-1, :] if grt2 is not None else None

            if cfg.AUG.MIRROR:
                if np.random.randint(0, 2) == 1:
                    img1 = img1[:, ::-1, :]
                    img2 = img2[:, ::-1, :] if img2 is not None else None
                    grt1 = grt1[:, ::-1]
                    grt2 = grt2[:, ::-1] if grt2 is not None else None

        elif ModelPhase.is_eval(mode):
            img1, img2, grt1, grt2 = aug.resize(
                img1, img2, grt1, grt2, mode=mode)
            img1, img2, grt1, grt2 = aug.rand_crop(
                img1, img2, grt1, grt2, mode=mode)
            if cfg.TEST.TEST_AUG:
                img1 = self.test_aug(img1)
                img2 = self.test_aug(img2) if img2 is not None else None

        elif ModelPhase.is_visual(mode):
            org_shape = [img1.shape[0], img1.shape[1]]
            img1, img2, grt1, grt2 = aug.resize(
                img1, img2, grt1, grt2, mode=mode)
            valid_shape = [img1.shape[0], img1.shape[1]]
            img1, img2, grt1, grt2 = aug.rand_crop(
                img1, img2, grt1, grt2, mode=mode)
        else:
            raise ValueError("Dataset mode={} Error!".format(mode))

        # Normalize image
        img1 = self.normalize_image(img1)
        img2 = self.normalize_image(img2) if img2 is not None else None

        if grt2 is not None:
            grt = grt1 * cfg.DATASET.NUM_CLASSES + grt2

            unchange_idx = np.where((grt1 - grt2) == 0)
            grt[unchange_idx] = 0
            if cfg.DATASET.NUM_CLASSES == 2:
                grt[np.where(grt != 0)] = 1

            ignore_idx = np.where((grt1 == cfg.DATASET.IGNORE_INDEX)
                                  | (grt2 == cfg.DATASET.IGNORE_INDEX))
            grt[ignore_idx] = cfg.DATASET.IGNORE_INDEX
        else:
            grt = grt1

        if ModelPhase.is_train(mode) or ModelPhase.is_eval(mode):
            grt = np.expand_dims(np.array(grt).astype('int32'), axis=0)
            ignore = (grt != cfg.DATASET.IGNORE_INDEX).astype('int32')

        if cfg.DATASET.INPUT_IMAGE_NUM == 1:
            if ModelPhase.is_train(mode):
                return (img1, grt, ignore)
            elif ModelPhase.is_eval(mode):
                return (img1, grt, ignore)
            elif ModelPhase.is_visual(mode):
                return (img1, grt, img1_name, valid_shape, org_shape)
        else:
            if ModelPhase.is_train(mode):
                return (img1, img2, grt, ignore)
            elif ModelPhase.is_eval(mode):
                return (img1, img2, grt, ignore)
            elif ModelPhase.is_visual(mode):
                return (img1, img2, grt, img1_name, img2_name, valid_shape,
                        org_shape)

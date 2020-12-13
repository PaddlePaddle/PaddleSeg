# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import random

import paddle
import numpy as np
from PIL import Image

from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class CityscapesAutolabeling(paddle.io.Dataset):
    """
    TODO: rewrite the comments
    """

    def __init__(self, transforms, dataset_root, mode='train',
                 coarse_proba=0.5):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = 19
        self.ignore_index = 255
        self.coarse_proba = coarse_proba

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        img_dir = os.path.join(self.dataset_root, 'leftImg8bit')
        label_dir = os.path.join(self.dataset_root, 'gtFine')
        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(
                    img_dir) or not os.path.isdir(label_dir):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        label_files = sorted(
            glob.glob(
                os.path.join(label_dir, mode, '*',
                             '*_gtFine_labelTrainIds.png')))
        img_files = sorted(
            glob.glob(os.path.join(img_dir, mode, '*', '*_leftImg8bit.png')))

        self.file_list = [[
            img_path, label_path
        ] for img_path, label_path in zip(img_files, label_files)]
        random.shuffle(self.file_list)

        if mode == 'train':

            # use coarse dataset only in training
            img_dir = os.path.join(self.dataset_root, 'leftImg8bit_trainextra',
                                   'leftImg8bit', 'train_extra')
            label_dir = os.path.join(self.dataset_root, 'convert_autolabelled')

            if self.dataset_root is None or not os.path.isdir(
                    self.dataset_root) or not os.path.isdir(
                        img_dir) or not os.path.isdir(label_dir):
                raise ValueError(
                    "The coarse dataset is not Found or the folder structure is nonconfoumance."
                )

            coarse_label_files = sorted(
                glob.glob(os.path.join(label_dir, '*', '*_leftImg8bit.png')))
            coarse_img_files = sorted(
                glob.glob(os.path.join(img_dir, '*', '*_leftImg8bit.png')))
            if len(coarse_img_files) != len(coarse_label_files):
                raise ValueError(
                    "The number of images = {} is not equal to the number of labels = {} in Cityscapes Autolabeling dataset."
                    .format(len(coarse_img_files), len(coarse_label_files)))

            self.coarse_file_list = [[img_path, label_path]
                                     for img_path, label_path in zip(
                                         coarse_img_files, coarse_label_files)]
            random.shuffle(self.coarse_file_list)
            self.num_coarse = len(self.coarse_file_list)
            self.rank = 0
            self.coarse_rank = 0

        # Keep the same number of files in one epoch even using coarse data.
        self.num_files = len(self.file_list)

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        if self.mode == 'test':
            im, _ = self.transforms(im=image_path)
            im = im[np.newaxis, ...]
            return im, image_path
        elif self.mode == 'val':
            im, _ = self.transforms(im=image_path)
            label = np.asarray(Image.open(label_path))
            label = label[np.newaxis, :, :]
            return im, label
        else:
            if idx / self.num_files < self.coarse_proba:
                #                 rand_idx = np.random.randint(0, len(self.coarse_file_list))
                #                 image_path, label_path = self.coarse_file_list[rand_idx]
                image_path, label_path = self.coarse_file_list[self.coarse_rank]
                self.coarse_rank += 1
                if self.coarse_rank >= self.num_coarse:
                    self.coarse_rank = 0
                    random.shuffle(self.coarse_file_list)
            else:
                #                 rand_idx = np.random.randint(0, len(self.file_list))
                #                 image_path, label_path = self.file_list[rand_idx]
                image_path, label_path = self.file_list[self.rank]
                self.rank += 1
                if self.rank >= self.num_files:
                    self.rank = 0
                    random.shuffle(self.file_list)
            im, label = self.transforms(im=image_path, label=label_path)
            return im, label

    def __len__(self):
        return self.num_files

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

# Random seed is set to ensure that after shuffling dataset per epoch during multi-gpu training, the data sequences of all gpus are consistent.
random.seed(100)


@manager.DATASETS.add_component
class CityscapesAutolabeling(paddle.io.Dataset):
    """
    Cityscapes dataset with fine data, coarse data and autolabelled data.
    Source: https://www.cityscapes-dataset.com/
    Autolabelled-Data from [google drive](https://drive.google.com/file/d/1DtPo-WP-hjaOwsbj6ZxTtOo_7R_4TKRG/view?usp=sharing)

    The folder structure is as follow:

        cityscapes
        |
        |--leftImg8bit
        |  |--train
        |  |--val
        |  |--test
        |
        |--gtFine
        |  |--train
        |  |--val
        |  |--test
        |
        |--leftImg8bit_trainextra
        |  |--leftImg8bit
        |     |--train_extra
        |        |--augsburg
        |        |--bayreuth
        |        |--...
        |
        |--convert_autolabelled
        |  |--augsburg
        |  |--bayreuth
        |  |--...

    Make sure there are **labelTrainIds.png in gtFine directory. If not, please run the conver_cityscapes.py in tools.
    Convert autolabelled data according to PaddleSeg data format:
        python tools/convert_cityscapes_autolabeling.py --dataset_root data/cityscapes/

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        coarse_multiple (float|int, optional): Multiple of the amount of coarse data relative to fine data. Default: 1
        add_val (bool, optional): Whether to add val set in training. Default: False
    """

    def __init__(self,
                 transforms,
                 dataset_root,
                 mode='train',
                 coarse_multiple=1,
                 add_val=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = 19
        self.ignore_index = 255
        self.coarse_multiple = coarse_multiple

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

        self.file_list = [
            [img_path, label_path]
            for img_path, label_path in zip(img_files, label_files)
        ]
        self.num_files = len(self.file_list)
        self.total_num_files = self.num_files

        if mode == 'train':
            # whether to add val set in training
            if add_val:
                label_files = sorted(
                    glob.glob(
                        os.path.join(label_dir, 'val', '*',
                                     '*_gtFine_labelTrainIds.png')))
                img_files = sorted(
                    glob.glob(
                        os.path.join(img_dir, 'val', '*', '*_leftImg8bit.png')))
                val_file_list = [
                    [img_path, label_path]
                    for img_path, label_path in zip(img_files, label_files)
                ]
                self.file_list.extend(val_file_list)
                self.num_files = len(self.file_list)

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

            self.total_num_files = int(self.num_files * (1 + coarse_multiple))

    def __getitem__(self, idx):
        if self.mode == 'test':
            image_path, label_path = self.file_list[idx]
            im, _ = self.transforms(im=image_path)
            im = im[np.newaxis, ...]
            return im, image_path
        elif self.mode == 'val':
            image_path, label_path = self.file_list[idx]
            im, _ = self.transforms(im=image_path)
            label = np.asarray(Image.open(label_path))
            label = label[np.newaxis, :, :]
            return im, label
        else:
            if idx >= self.num_files:
                image_path, label_path = self.coarse_file_list[idx -
                                                               self.num_files]
            else:
                image_path, label_path = self.file_list[idx]

            im, label = self.transforms(im=image_path, label=label_path)
            return im, label

    def shuffle(self):
        random.shuffle(self.coarse_file_list)

    def __len__(self):
        return self.total_num_files

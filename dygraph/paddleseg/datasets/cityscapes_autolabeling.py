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

from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class CityscapesAutolabeling(Dataset):
    """
    Cityscapes dataset `https://www.cityscapes-dataset.com/`.
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

    Make sure there are **labelTrainIds.png in gtFine directory. If not, please run the conver_cityscapes.py in tools.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
    """

    def __init__(self,
                 transforms,
                 dataset_root,
                 mode='train',
                 autolabeling_percentage=0.5):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = 19
        self.ignore_index = 255

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

        if mode == 'train':
            random.shuffle(self.file_list)

            total_num = len(self.file_list)
            autolabeling_num = int(total_num * autolabeling_percentage)
            origin_num = total_num - autolabeling_num

            self.file_list = self.file_list[:origin_num]

            # count autolabeling
            img_dir = os.path.join(self.dataset_root, 'leftImg8bit_trainextra',
                                   'leftImg8bit', 'train_extra')
            label_dir = os.path.join(self.dataset_root, 'convert_autolabelled')
            # label_dir = os.path.join(self.dataset_root, 'autolabelled',
            #                          'train_extra')
            if self.dataset_root is None or not os.path.isdir(
                    self.dataset_root) or not os.path.isdir(
                        img_dir) or not os.path.isdir(label_dir):
                raise ValueError(
                    "The dataset is not Found or the folder structure is nonconfoumance."
                )

            autolabeling_label_files = sorted(
                glob.glob(os.path.join(label_dir, '*', '*_leftImg8bit.png')))
            autolabeling_img_files = sorted(
                glob.glob(os.path.join(img_dir, '*', '*_leftImg8bit.png')))
            if len(autolabeling_img_files) != len(autolabeling_label_files):
                raise ValueError(
                    "The number of images = {} is not equal to the number of labels = {} in Cityscapes Autolabeling dataset."
                    .format(
                        len(autolabeling_img_files),
                        len(autolabeling_label_files)))

            autolabeling_file_list = [
                [img_path, label_path] for img_path, label_path in zip(
                    autolabeling_img_files, autolabeling_label_files)
            ]
            random.shuffle(autolabeling_file_list)
            autolabeling_file_list = autolabeling_file_list[:autolabeling_num]

            self.file_list.extend(autolabeling_file_list)
        pass

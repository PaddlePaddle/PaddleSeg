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

import numpy as np
import paddle

from paddleseg.cvlibs import manager
from paddleseg.transforms.transforms import Compose
from transforms.lane_transforms import LaneRandomRotation, SubImgCrop


@manager.DATASETS.add_component
class Tusimple(paddle.io.Dataset):
    """
    Tusimple dataset `https://github.com/TuSimple/tusimple-benchmark/issues/3`.
    The folder structure is as follow:
     |-- tusimple
        |-- train_set
            |-- clips
                |-- 0313-1
                |-- 0313-2
                |-- 0531
                |-- 0601
            |-- labels [need to generate label dir]
                |-- 0313-1
                |-- 0313-2
                |-- 0531
                |-- 0601
            |-- train_list.txt [need to generate]
            |-- label_data_0313.json
            |-- label_data_0531.json
            |-- label_data_0601.json
        |-- test_set
            |-- clips
                |-- 0530
                |-- 0531
                |-- 0601
            |-- labels [need to generate label dir]
                |-- 0530
                |-- 0531
                |-- 0601
            |-- train_list.txt [need to generate]
            |-- test_tasks_0627.json
            |-- test_label.json

    Make sure test_label.json is in test_set directory, and there are labels directory in train_set and test_set directory.
    If not, please run the generate_seg_tusimple.py in utils.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): tusimple dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        cut_height (int, optional): Whether to cut image height while training. Default: 0
    """
    NUM_CLASSES = 7

    def __init__(self,
                 transforms=None,
                 dataset_root=None,
                 mode='train',
                 cut_height=0):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms, to_rgb=False)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.cut_height = cut_height
        self.test_gt_json = os.path.join(self.dataset_root,
                                         'test_set/test_label.json')

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "`mode` should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if self.dataset_root is None:
            raise ValueError("`dataset_root` is necessary, but it is None.")

        if mode == 'train':
            file_path = os.path.join(self.dataset_root,
                                     'train_set/train_list.txt')
        elif mode == 'val':
            file_path = os.path.join(self.dataset_root,
                                     'test_set/test_list.txt')
        else:
            file_path = os.path.join(self.dataset_root,
                                     'test_set/test_list.txt')

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split()
                if len(items) != 2:
                    if mode == 'train' or mode == 'val':
                        raise Exception(
                            "File list format incorrect! It should be"
                            " image_name label_name\\n")
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = None
                else:
                    image_path = self.dataset_root + items[0]
                    label_path = self.dataset_root + items[1]
                self.file_list.append([image_path, label_path])

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]

        if self.mode == 'test':
            im, _ = self.transforms(im=image_path)
            im = im[np.newaxis, ...]
            return im, image_path

        elif self.mode == 'val':
            im, label = self.transforms(im=image_path, label=label_path)
            return im, label, image_path
        else:
            im, label = self.transforms(im=image_path, label=label_path)
            return im, label

    def __len__(self):
        return len(self.file_list)

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
class MixedDataset(paddle.io.Dataset):
    """
    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        data_ratio (float|int, optional): Multiple of the amount of coarse data relative to fine data. Default: 1
    """

    def __init__(self,
                 train_datasets,
                 train_data_ratio,
                 transforms=None,
                 num_classes=2):
        self.train_datasets = train_datasets
        if transforms is None:
            transforms = train_datasets[0].transforms
        self.transforms = Compose(transforms)
        self.all_file_list = list()
        self.file_lists = list()
        self.num_classes = num_classes
        self.ignore_index = 255
        self.train_data_ratio = train_data_ratio
        self.separator = ' '
        self.mode = 'train'

        for i, train_dataset in enumerate(train_datasets):
            file_list = train_dataset.file_list
            random.shuffle(file_list)
            self.file_lists.append(file_list)
            self.all_file_list = create_file_set(self.all_file_list, file_list,
                                                 self.train_data_ratio[i],
                                                 self.separator, self.mode)
        random.shuffle(self.all_file_list)
        self.total_file_num = len(self.all_file_list)

    def __getitem__(self, idx):
        data = {}
        data['trans_info'] = []
        image_path, label_path = self.all_file_list[idx]
        data['img'] = image_path
        data['label'] = label_path
        # If key in gt_fields, the data[key] have transforms synchronous.
        data['gt_fields'] = []

        if self.mode == 'val':
            data = self.transforms(data)
            data['label'] = data['label'][np.newaxis, :, :]

        else:
            data['gt_fields'].append('label')
            data = self.transforms(data)
        return data

    def shuffle(self):
        self.all_file_list = list()
        for i, file_list in enumerate(self.file_lists):
            random.shuffle(file_list)
            self.all_file_list = create_file_set(self.all_file_list, file_list,
                                                 self.train_data_ratio[i],
                                                 self.separator, self.mode)
        random.shuffle(self.all_file_list)

    def __len__(self):
        return self.total_file_num


def create_file_set(all_file_list, file_list, data_ratio, separator, mode):
    file_num = len(file_list)
    if data_ratio == 1:
        all_file_list.extend(file_list)
    elif data_ratio < 1:
        all_file_list.extend(file_list[:int(data_ratio * file_num)])
    else:
        if isinstance(data_ratio, float):
            for j in range(int(data_ratio)):
                all_file_list.extend(file_list)
            ratio = data_ratio - int(data_ratio)
            all_file_list.extend(file_list[:int(ratio * file_num)])
        else:
            for j in range(data_ratio):
                all_file_list.extend(file_list)
    return all_file_list


def read_file_list(file_path, separator, mode, dataset_root):
    file_list = list()
    with open(file_path, 'r') as f:
        for line in f:
            items = line.strip().split(separator)
            if len(items) != 2:
                if mode == 'train' or mode == 'val':
                    raise ValueError(
                        "File list format incorrect in {}! In training or evaluation task it should be"
                        " image_name{}label_name\\n".format(file_path,
                                                            separator))
                image_path = os.path.join(dataset_root, items[0])
                label_path = None
            else:
                image_path = os.path.join(dataset_root, items[0])
                label_path = os.path.join(dataset_root, items[1])
            file_list.append([image_path, label_path])
    return file_list

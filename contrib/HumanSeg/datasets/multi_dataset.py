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
class MultiDataset(paddle.io.Dataset):
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
        data_ratio (float|int, optional): Multiple of the amount of coarse data relative to fine data. Default: 1
    """

    def __init__(self,
                 transforms,
                 dataset_root_list,
                 valset_weight,
                 valset_class_weight,
                 data_ratio,
                 mode=None,
                 file_list_name=None,
                 num_classes=2,
                 val_test_set_rank=0):
        self.dataset_root_list = dataset_root_list
        self.transforms = Compose(transforms)
        self.all_file_list = list()
        self.file_lists = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = num_classes
        self.ignore_index = 255
        self.data_ratio = data_ratio
        self.valset_weight = valset_weight
        self.valset_class_weight = valset_class_weight
        self.separator = ' '

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if mode == 'train':

            for i, dataset_root in enumerate(dataset_root_list):
                if not os.path.exists(dataset_root):
                    raise FileNotFoundError(
                        'there is not `dataset_root`: {}.'.format(dataset_root))

                train_path = os.path.join(dataset_root, file_list_name)
                if not os.path.exists(train_path):
                    raise FileNotFoundError(
                        '`train_path` is not found: {}'.format(train_path))

                file_list = read_file_list(train_path, self.separator, mode,
                                           dataset_root)
                random.shuffle(file_list)
                self.file_lists.append(file_list)
                self.all_file_list = create_file_set(
                    self.all_file_list, file_list, self.data_ratio[i],
                    self.separator, self.mode)
            random.shuffle(self.all_file_list)
            self.total_file_num = len(self.all_file_list)
        elif mode == 'val':
            val_path = os.path.join(dataset_root_list[val_test_set_rank],
                                    file_list_name)
            if not os.path.exists(val_path):
                raise FileNotFoundError(
                    '`val_path` is not found: {}'.format(val_path))
            self.file_list = read_file_list(
                val_path, self.separator, mode,
                dataset_root_list[val_test_set_rank])
            self.total_file_num = len(self.file_list)
        else:
            test_path = os.path.join(dataset_root_list[val_test_set_rank],
                                     file_list_name)
            if not os.path.exists(test_path):
                raise FileNotFoundError(
                    '`test_path` is not found: {}'.format(test_path))
            self.file_list = read_file_list(
                test_path, self.separator, mode,
                dataset_root_list[val_test_set_rank])
            self.total_file_num = len(self.file_list)

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
            image_path, label_path = self.all_file_list[idx]
            im, label = self.transforms(im=image_path, label=label_path)
            return im, label

    def shuffle(self):
        self.all_file_list = list()
        for i, file_list in enumerate(self.file_lists):
            random.shuffle(file_list)
            self.all_file_list = create_file_set(self.all_file_list, file_list,
                                                 self.data_ratio[i],
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
                        "File list format incorrect! In training or evaluation task it should be"
                        " image_name{}label_name\\n".format(separator))
                image_path = os.path.join(dataset_root, items[0])
                label_path = None
            else:
                image_path = os.path.join(dataset_root, items[0])
                label_path = os.path.join(dataset_root, items[1])
            file_list.append([image_path, label_path])
    return file_list

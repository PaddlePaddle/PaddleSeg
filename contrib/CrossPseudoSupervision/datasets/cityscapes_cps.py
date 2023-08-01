# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddleseg.transforms import Compose

from cvlibs import manager
from .basedataset import BaseDataset


@manager.DATASETS.add_component
class CityscapesCPS(BaseDataset):
    """
    Semi-supervision Cityscapes dataset with images, segmentation labels and data list.
    Source: https://www.cityscapes-dataset.com/
    Semi-supervision Cityscapes dataset from [OneDrive](https://pkueducn-my.sharepoint.com/:f:/g/personal/pkucxk_pku_edu_cn/EtjNKU0oVMhPkOKf9HTPlVsBIHYbACel6LSvcUeP4MXWVg?e=139icd)

    The folder structure is as follows:

        city
        ├── config_new
        │    ├── coarse_split
        │    │   ├── train_extra_3000.txt
        │    │   ├── train_extra_6000.txt
        │    │   └── train_extra_9000.txt
        │    ├── subset_train
        │    │   ├── train_aug_labeled_1-16.txt
        │    │   ├── train_aug_labeled_1-2.txt
        │    │   ├── train_aug_labeled_1-4.txt
        │    │   ├── train_aug_labeled_1-8.txt
        │    │   ├── train_aug_unlabeled_1-16.txt
        │    │   ├── train_aug_unlabeled_1-2.txt
        │    │   ├── train_aug_unlabeled_1-4.txt
        │    │   └── train_aug_unlabeled_1-8.txt
        │    ├── test.txt
        │    ├── train.txt
        │    ├── train_val.txt
        │    └── val.txt  
        ├── generate_colored_gt.py
        ├── images
        │   ├── test
        │   ├── train
        │   └── val
        └── segmentation
            ├── test
            ├── train
            └── val

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        unsupervised (bool, optional): Whether haven't labels. Default: False.
    """
    NUM_CLASSES = 19
    IGNORE_INDEX = 255
    IMG_CHANNELS = 3

    def __init__(self,
                 mode,
                 dataset_root,
                 transforms,
                 unsupervised=False,
                 train_path=None,
                 val_path=None,
                 file_length=None,
                 img_channels=3,
                 ignore_index=255,
                 separator='\t'):
        super(CityscapesCPS, self).__init__(
            mode, dataset_root, transforms, train_path, val_path, file_length,
            img_channels, ignore_index, separator)

        self.unsupervised = unsupervised

        self.num_classes = self.NUM_CLASSES

        if self.mode not in ['train', 'val']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    self.mode))
        self._file_names = self._get_file_names(mode)

        if not os.path.exists(dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                dataset_root))
        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

    def __getitem__(self, index):
        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]

        image_path = self.dataset_root + names[0]
        label_path = self.dataset_root + names[1]

        if not self.unsupervised and self.mode == 'train':
            data = {'img': image_path, 'label': label_path}
            data['gt_fields'] = ['label']
            data['trans_info'] = []
            data = self.transforms(data)

        elif not self.unsupervised and self.mode == 'val':
            data = {'img': image_path, 'label': label_path}
            data['gt_fields'] = []
            data['trans_info'] = []
            data = self.transforms(data)
            data['label'] = data['label'][np.newaxis, :, :]
        elif self.unsupervised and self.mode == 'train':
            data = {'img': image_path}
            data['trans_info'] = []
            data['label'] = []
            data = self.transforms(data)
        else:
            raise ValueError(
                "when mode is 'val', self.unsupervised should be False.")

        return data

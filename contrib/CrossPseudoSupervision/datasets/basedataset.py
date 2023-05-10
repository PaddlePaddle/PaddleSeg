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

import paddle
import numpy as np

from paddle.io import Dataset
from paddleseg.transforms import Compose


class BaseDataset(Dataset):
    """
    Pass in a custom dataset that conforms to the format.

    Args:
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory.
        train_path (str, optional): The train dataset file. When mode is 'train', train_path is necessary.
            The contents of train_path file are as follow:
            image1.jpg ground_truth1.png
            image2.jpg ground_truth2.png
        val_path (str, optional): The evaluation dataset file. When mode is 'val', val_path is necessary.
            The contents is the same as train_path
        file_length (int, optional): The number of trainable data in the dataset. Default: None.
        ignore_index (int, optional): The category that needs to be ignored during the training process.
        separator (str, optional): The separator of dataset list.
    """

    def __init__(self,
                 mode,
                 dataset_root,
                 transforms,
                 train_path=None,
                 val_path=None,
                 file_length=None,
                 img_channels=3,
                 ignore_index=255,
                 separator='\t'):
        super(BaseDataset, self).__init__()
        self.mode = mode.lower()
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self._train_source = train_path
        self._eval_source = val_path
        self._file_length = file_length
        self.ignore_index = ignore_index
        self.separator = separator
        self.img_channels = img_channels

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

        if img_channels not in [1, 3]:
            raise ValueError("`img_channels` should in [1, 3], but got {}".
                             format(img_channels))

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]
        image_path = os.path.join(self.dataset_root, names[0])
        label_path = os.path.join(self.dataset_root, names[1])
        data = {}
        data['trans_info'] = []
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

    def _get_file_names(self, split_name, train_extra=False):
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            img_name, gt_name = self._process_item_names(item, self.separator)
            file_names.append([img_name, gt_name])

        if train_extra:
            file_names2 = []
            source2 = self._train_source.replace('train', 'train_extra')
            with open(source2) as f:
                files2 = f.readlines()

            for item in files2:
                img_name, gt_name = self._process_item_names(item,
                                                             self.separator)
                file_names2.append([img_name, gt_name])

            return file_names, file_names2

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)

        if length < files_len:
            return self._file_names[:length]

        new_file_names = self._file_names * (length // files_len)
        rand_indices = paddle.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    @staticmethod
    def _process_item_names(item, separator='\t'):
        item = item.strip()
        item = item.split(separator)
        img_name = item[0]

        if len(item) == 1:
            gt_name = None
        else:
            gt_name = item[1]

        return img_name, gt_name

    def get_length(self):
        return self.__len__()

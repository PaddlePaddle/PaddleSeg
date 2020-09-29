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

import paddle
import numpy as np
from PIL import Image

from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class Dataset(paddle.io.Dataset):
    """Pass in a custom dataset that conforms to the format.

    Args:
        dataset_root: The dataset directory.
        num_classes: Number of classes.
        mode: which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        train_list: The train dataset file. When image_set is 'train', train_list is necessary.
            The contents of train_list file are as follow:
            image1.jpg ground_truth1.png
            image2.jpg ground_truth2.png
        val_list: The evaluation dataset file. When image_set is 'val', val_list is necessary.
            The contents is the same as train_list
        test_list: The test dataset file. When image_set is 'test', test_list is necessary.
            The annotation file is not necessary in test_list file.
        separator: The separator of dataset list. Default: ' '.
        transforms: Transforms for image.

        Examples:
            todo

    """

    def __init__(self,
                 dataset_root,
                 num_classes,
                 mode='train',
                 train_list=None,
                 val_list=None,
                 test_list=None,
                 separator=' ',
                 transforms=None,
                 ignore_index=255):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        self.mode = mode
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        if mode.lower() not in ['train', 'val', 'test']:
            raise Exception(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise Exception("`transforms` is necessary, but it is None.")

        self.dataset_root = dataset_root
        if not os.path.exists(self.dataset_root):
            raise Exception('there is not `dataset_root`: {}.'.format(
                self.dataset_root))

        if mode == 'train':
            if train_list is None:
                raise Exception(
                    'When `mode` is "train", `train_list` is necessary, but it is None.'
                )
            elif not os.path.exists(train_list):
                raise Exception(
                    '`train_list` is not found: {}'.format(train_list))
            else:
                file_list = train_list
        elif mode == 'val':
            if val_list is None:
                raise Exception(
                    'When `mode` is "val", `val_list` is necessary, but it is None.'
                )
            elif not os.path.exists(val_list):
                raise Exception('`val_list` is not found: {}'.format(val_list))
            else:
                file_list = val_list
        else:
            if test_list is None:
                raise Exception(
                    'When `mode` is "test", `test_list` is necessary, but it is None.'
                )
            elif not os.path.exists(test_list):
                raise Exception(
                    '`test_list` is not found: {}'.format(test_list))
            else:
                file_list = test_list

        with open(file_list, 'r') as f:
            for line in f:
                items = line.strip().split(separator)
                if len(items) != 2:
                    if mode == 'train' or mode == 'val':
                        raise Exception(
                            "File list format incorrect! In training or evaluation task it should be"
                            " image_name{}label_name\\n".format(separator))
                    image_path = os.path.join(self.dataset_root, items[0])
                    grt_path = None
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    grt_path = os.path.join(self.dataset_root, items[1])
                self.file_list.append([image_path, grt_path])

    def __getitem__(self, idx):
        image_path, grt_path = self.file_list[idx]
        if self.mode == 'test':
            im, im_info, _ = self.transforms(im=image_path)
            im = im[np.newaxis, ...]
            return im, im_info, image_path
        elif self.mode == 'val':
            im, im_info, _ = self.transforms(im=image_path)
            im = im[np.newaxis, ...]
            label = np.asarray(Image.open(grt_path))
            label = label[np.newaxis, np.newaxis, :, :]
            return im, im_info, label
        else:
            im, im_info, label = self.transforms(im=image_path, label=grt_path)
            return im, label

    def __len__(self):
        return len(self.file_list)

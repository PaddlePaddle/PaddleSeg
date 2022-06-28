# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from medicalseg.cvlibs import manager
from medicalseg.transforms import Compose
from medicalseg.utils.env_util import seg_env
import medicalseg.transforms.functional as F
from medicalseg.utils.download import download_file_and_uncompress


@manager.DATASETS.add_component
class MedicalDataset(paddle.io.Dataset):
    """
    Pass in a custom dataset that conforms to the format.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory.
        num_classes (int): Number of classes.
        result_dir (str): The directory to save the next phase result.
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        ignore_index (int, optional): The index that ignore when calculate loss.

        Examples:

            import medicalseg.transforms as T
            from paddleseg.datasets import MedicalDataset

            transforms = [T.RandomRotation3D(degrees=90)]
            dataset_root = 'dataset_root_path'
            dataset = MedicalDataset(transforms = transforms,
                              dataset_root = dataset_root,
                              num_classes = 3,
                              mode = 'train')

            for data in dataset:
                img, label = data
                print(img.shape, label.shape)
                print(np.unique(label))

    """

    def __init__(self,
                 dataset_root,
                 result_dir,
                 transforms,
                 num_classes,
                 mode='train',
                 ignore_index=255,
                 data_URL="",
                 dataset_json_path=""):
        self.dataset_root = dataset_root
        self.result_dir = result_dir
        self.transforms = Compose(transforms)
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.ignore_index = ignore_index  # todo: if labels only have 1/0/2, ignore_index is not necessary
        self.dataset_json_path = dataset_json_path

        if self.dataset_root is None:
            self.dataset_root = download_file_and_uncompress(
                url=data_URL,
                savepath=seg_env.DATA_HOME,
                extrapath=seg_env.DATA_HOME)
        elif not os.path.exists(self.dataset_root):
            raise ValueError(
                "The `dataset_root` don't exist please specify the correct path to data."
            )

        if mode == 'train':
            file_path = os.path.join(self.dataset_root, 'train_list.txt')
        elif mode == 'val':
            file_path = os.path.join(self.dataset_root, 'val_list.txt')
        elif mode == 'test':
            file_path = os.path.join(self.dataset_root, 'test_list.txt')
        else:
            raise ValueError(
                "`mode` should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split()
                if len(items) != 2:
                    raise Exception("File list format incorrect! It should be"
                                    " image_name label_name\\n")
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    grt_path = os.path.join(self.dataset_root, items[1])
                self.file_list.append([image_path, grt_path])

        if mode == 'train':
            self.file_list = self.file_list * 10

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]

        im, label = self.transforms(im=image_path, label=label_path)

        return im, label, self.file_list[idx][0]  # npy file name

    def save_transformed(self):
        """Save the preprocessed images to the result_dir"""
        pass  # todo

    def __len__(self):
        return len(self.file_list)

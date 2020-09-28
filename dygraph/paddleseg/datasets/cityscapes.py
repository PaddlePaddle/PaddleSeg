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

from .dataset import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class Cityscapes(Dataset):
    """Cityscapes dataset `https://www.cityscapes-dataset.com/`.
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
        dataset_root: Cityscapes dataset directory.
        mode: Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        transforms: Transforms for image.
    """

    def __init__(self, dataset_root, transforms=None, mode='train'):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        self.mode = mode
        self.num_classes = 19
        self.ignore_index = 255

        if mode.lower() not in ['train', 'val', 'test']:
            raise Exception(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise Exception("`transforms` is necessary, but it is None.")

        img_dir = os.path.join(self.dataset_root, 'leftImg8bit')
        grt_dir = os.path.join(self.dataset_root, 'gtFine')
        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(
                    img_dir) or not os.path.isdir(grt_dir):
            raise Exception(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        grt_files = sorted(
            glob.glob(
                os.path.join(grt_dir, mode, '*', '*_gtFine_labelTrainIds.png')))
        img_files = sorted(
            glob.glob(os.path.join(img_dir, mode, '*', '*_leftImg8bit.png')))

        self.file_list = [[img_path, grt_path]
                          for img_path, grt_path in zip(img_files, grt_files)]

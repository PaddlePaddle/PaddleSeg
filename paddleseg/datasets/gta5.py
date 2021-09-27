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

from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class GTA5(Dataset):
    """
    Cityscapes dataset `https://download.visinf.tu-darmstadt.de/data/from_games/`.
    The folder structure is as follow:

        cityscapes
        |
        |--images
        |
        |--labels

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 19

    def __init__(self, transforms, dataset_root, mode='train', edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        img_dir = os.path.join(self.dataset_root, 'images')
        label_dir = os.path.join(self.dataset_root, 'labels')
        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(
                    img_dir) or not os.path.isdir(label_dir):
            print(
                os.path.isdir(img_dir), os.path.isdir(label_dir),
                os.path.isdir(self.dataset_root))
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )
        modelist_path = os.path.join(self.dataset_root, 'gta5_list',
                                     mode + '.txt')
        items = [int(imgid.strip()) for imgid in open(modelist_path)]

        img_files = sorted(
            [os.path.join(img_dir, f"{imgid:0>5d}.png") for imgid in items])
        label_files = sorted([
            os.path.join(label_dir, f"{imgid:0>5d}color19.png")
            for imgid in items
        ])

        self.file_list = [[
            img_path, label_path
        ] for img_path, label_path in zip(img_files, label_files)]


if __name__ == '__main__':
    import paddleseg
    transforms = [paddleseg.transforms.Normalize()]
    trainset = GTA5(transforms, '/ssd1/tangshiyu/data/GTA5')
    print(trainset.file_list[0], trainset[0])
    # ['/ssd1/tangshiyu/data/GTA5/images/00001.png', '/ssd1/tangshiyu/data/GTA5/labels/00001color19.png']

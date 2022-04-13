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

import numpy as np

from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class ImagenetPseudoSeg(Dataset):
    """
    The imagenet pseudo dataset for segmentation.
    The folder structure is as follow:

        dataset_root
        |
        |--ILSVRC2012
        |  |--train
        |
        |--imagenet_pseudo_sg_v2
        |  |--train
        |  |--train1M.txt
        |  |--imagenet_lsvrc_2015_synsets.txt

    Args:
        dataset_root (str): Cityscapes dataset directory.
        transforms (list): Transforms for image.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    ignore_index = 1001  # 0~999 is target class, 1000 is bg
    NUM_CLASSES = 1001  # consider target class and bg

    def __init__(self,
                 dataset_root,
                 transforms,
                 mode='train',
                 edge=False,
                 imagenet_dirname="ILSVRC2012",
                 pseudo_seg_dirname="imagenet_pseudo_sg_v2"):
        mode = mode.lower()
        if mode not in ['train']:
            raise ValueError("mode should be 'train', but got {}.".format(mode))
        if transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        self.transforms = Compose(transforms)
        self.mode = mode
        self.edge = edge

        self.num_classes = self.NUM_CLASSES
        self.ignore_index = self.num_classes  # 1001
        self.file_list = []
        self.class_id_dict = {}

        img_dir = os.path.join(dataset_root, imagenet_dirname)
        label_dir = os.path.join(dataset_root, pseudo_seg_dirname)
        if dataset_root is None or not os.path.isdir(dataset_root) \
            or not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        train_list_file = os.path.join(label_dir, "train1M.txt")
        if not os.path.exists(train_list_file):
            raise ValueError("Train list file isn't exists.")
        for idx, line in enumerate(open(train_list_file)):
            label_path = line.strip()
            img_path = label_path[0:-4]
            label_path = os.path.join(label_dir, label_path)
            img_path = os.path.join(img_dir, img_path)
            self.file_list.append([img_path, label_path])

        class_id_file = os.path.join(label_dir,
                                     "imagenet_lsvrc_2015_synsets.txt")
        if not os.path.exists(class_id_file):
            raise ValueError("Class id file isn't exists.")
        for idx, line in enumerate(open(class_id_file)):
            class_type = line.strip()
            self.class_id_dict[class_type] = idx

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        im, label = self.transforms(im=image_path, label=label_path)

        class_type = (image_path.split('/')[-1]).split('_')[0]
        class_id = self.class_id_dict[class_type]
        label = label.astype("int64")
        label[label == 255] = self.ignore_index
        label[label <
              9] = 1000  # [0, 999] for imagenet classes, 1000 for background.
        label[label == 9] = class_id

        if self.edge:
            edge_mask = F.mask_to_binary_edge(
                label, radius=2, num_classes=self.num_classes)
            return im, label, edge_mask
        else:
            return im, label

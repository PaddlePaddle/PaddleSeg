# This file is made available under Apache License, Version 2.0
# This file is based on code available under the MIT License here:
#  https://github.com/ZJULearning/MaxSquareLoss/blob/master/datasets/GTA5Dataset.py
#
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from PIL import Image, ImageFile
from datasets.cityscapes_noconfig import CityDataset, to_tuple

import paddle
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GTA5Dataset(CityDataset):
    def __init__(self,
                 root='./datasets/GTA5',
                 list_path='./datasets/GTA5/list',
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True,
                 random_mirror=False,
                 random_crop=False,
                 resize=False,
                 gaussian_blur=False,
                 class_16=False,
                 edge=True):

        # Args
        self.data_path = root
        self.list_path = list_path
        self.split = split
        self.base_size = to_tuple(base_size)
        self.crop_size = to_tuple(crop_size)
        self.training = training
        self.class_16 = False
        self.class_13 = False
        self.NUM_CLASSES = 19
        self.ignore_label = 255
        self.edge = edge

        assert class_16 == False

        # Augmentation
        self.random_mirror = random_mirror
        self.random_crop = random_crop
        self.resize = resize
        self.gaussian_blur = gaussian_blur

        # Files
        item_list_filepath = os.path.join(self.list_path, self.split + ".txt")
        if not os.path.exists(item_list_filepath):
            print(item_list_filepath)
            raise Warning("split must be train/val/trainval/test/all")
        self.image_filepath = os.path.join(self.data_path, "images")
        self.gt_filepath = os.path.join(self.data_path, "labels")
        self.items = [id.strip() for id in open(item_list_filepath)]

        # Label map
        self.id_to_trainid = {
            7: 0,
            8: 1,
            11: 2,
            12: 3,
            13: 4,
            17: 5,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            31: 16,
            32: 17,
            33: 18
        }

        print("{} num images in GTA5 {} set have been loaded.".format(
            len(self.items), self.split))

    def __getitem__(self, item):
        idx = int(self.items[item])
        name = f"{idx:0>5d}.png"

        # Open image and label
        image_path = os.path.join(self.image_filepath, name)
        gt_image_path = os.path.join(self.gt_filepath, name)
        image = Image.open(image_path).convert("RGB")
        gt_image = Image.open(gt_image_path)

        # Augmentation
        if (self.split == "train" or self.split == "trainval"
                or self.split == "all") and self.training:
            image, gt_image, edge_mask = self._train_sync_transform(
                image, gt_image)
        else:
            image, gt_image, edge_mask = self._val_sync_transform(
                image, gt_image)

        return image, gt_image, edge_mask

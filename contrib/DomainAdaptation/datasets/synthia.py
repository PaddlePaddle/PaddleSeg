# This file is made available under Apache License, Version 2.0
# This file is based on code available under the MIT License here:
#  https://github.com/ZJULearning/MaxSquareLoss/blob/master/datasets/SYNTHIADataset.py
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
import imageio
import numpy as np
from PIL import Image

from datasets.cityscapes_noconfig import CityDataset, to_tuple

imageio.plugins.freeimage.download()
synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]


class SYNTHIADataset(CityDataset):
    def __init__(
            self,
            root='./datasets/SYNTHIA',
            list_path='./datasets/SYNTHIA/list',
            split='train',
            base_size=769,
            crop_size=769,
            training=True,
            class_16=False,
            random_mirror=False,
            random_crop=False,
            resize=False,
            gaussian_blur=False,
    ):

        # Args
        self.data_path = root
        self.list_path = list_path
        self.split = split
        self.base_size = to_tuple(base_size)
        self.crop_size = to_tuple(crop_size)
        self.training = training

        # Augmentations
        self.random_mirror = random_mirror
        self.random_crop = random_crop
        self.resize = resize
        self.gaussian_blur = gaussian_blur

        # Files
        item_list_filepath = os.path.join(self.list_path, self.split + ".txt")
        if not os.path.exists(item_list_filepath):
            raise Warning("split must be train/val/trainavl/test")
        self.image_filepath = os.path.join(self.data_path, "RGB")
        self.gt_filepath = os.path.join(self.data_path, "GT/LABELS")
        self.items = [id.strip() for id in open(item_list_filepath)]

        # Label map
        self.id_to_trainid = {
            1: 10,
            2: 2,
            3: 0,
            4: 1,
            5: 4,
            6: 8,
            7: 5,
            8: 13,
            9: 7,
            10: 11,
            11: 18,
            12: 17,
            15: 6,
            16: 9,
            17: 12,
            18: 14,
            19: 15,
            20: 16,
            21: 3
        }

        # Only consider 16 shared classes
        self.class_16 = class_16
        self.trainid_to_16id = {id: i for i, id in enumerate(synthia_set_16)}
        self.class_13 = False

        print("{} num images in SYNTHIA {} set have been loaded.".format(
            len(self.items), self.split))

    def __getitem__(self, item):
        id = int(self.items[item])
        name = f"{id:0>7d}.png"

        # Open image and label
        image_path = os.path.join(self.image_filepath, name)
        gt_image_path = os.path.join(self.gt_filepath, name)
        image = Image.open(image_path).convert("RGB")
        gt_image = imageio.imread(gt_image_path, format='PNG-FI')[:, :, 0]
        gt_image = Image.fromarray(np.uint8(gt_image))

        # Augmentations
        if (self.split == "train" or self.split == "trainval"
                or self.split == "all") and self.training:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)
        return image, gt_image, item

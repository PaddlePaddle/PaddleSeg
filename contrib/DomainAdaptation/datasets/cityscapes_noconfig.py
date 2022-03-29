# This file is made available under Apache License, Version 2.0
# This file is based on code available under the MIT License here:
#  https://github.com/ZJULearning/MaxSquareLoss/blob/master/datasets/cityscapes_Dataset.py
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
import random
import numpy as np
import collections.abc as abc
from PIL import Image, ImageOps, ImageFilter, ImageFile

import paddle
from paddle import io

import paddleseg.transforms.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_MEAN = np.array(
    (104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

label_colours = list(
    map(
        tuple,
        [
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
            [0, 0, 0],  # the color of ignored label
        ]))

# Labels
ignore_label = 255
cityscapes_id_to_trainid = {
    -1: ignore_label,
    0: ignore_label,
    1: ignore_label,
    2: ignore_label,
    3: ignore_label,
    4: ignore_label,
    5: ignore_label,
    6: ignore_label,
    7: 0,
    8: 1,
    9: ignore_label,
    10: ignore_label,
    11: 2,
    12: 3,
    13: 4,
    14: ignore_label,
    15: ignore_label,
    16: ignore_label,
    17: 5,
    18: ignore_label,
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
    29: ignore_label,
    30: ignore_label,
    31: 16,
    32: 17,
    33: 18
}


def to_tuple(x):
    return x if isinstance(x, abc.Iterable) else (x, x)


class CityDataset(io.Dataset):
    def __init__(self,
                 root,
                 list_path,
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True,
                 random_mirror=False,
                 random_crop=False,
                 resize=False,
                 gaussian_blur=False,
                 class_16=False,
                 class_13=False,
                 edge=True,
                 logger=None):
        self.data_path = root
        self.list_path = list_path
        self.split = split
        self.base_size = to_tuple(base_size)
        self.crop_size = to_tuple(crop_size)
        self.training = training
        self.logger = logger

        # Augmentations
        self.random_mirror = random_mirror
        self.random_crop = random_crop
        self.resize = resize
        self.gaussian_blur = gaussian_blur
        self.NUM_CLASSES = 19
        self.ignore_label = 255
        self.edge = edge

        # Files
        item_list_filepath = os.path.join(self.list_path, self.split + ".txt")
        if not os.path.exists(item_list_filepath):
            raise Warning("split must be train/val/trainval")
        self.image_filepath = os.path.join(self.data_path, "leftImg8bit")
        self.gt_filepath = os.path.join(self.data_path, "gtFine")
        self.items = [id.strip() for id in open(item_list_filepath)]
        self.id_to_trainid = cityscapes_id_to_trainid

        # In SYNTHIA-to-Cityscapes case, only consider 16 shared classes
        self.class_16 = class_16
        synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_16id = {id: i for i, id in enumerate(synthia_set_16)}

        # In Cityscapes-to-NTHU case, only consider 13 shared classes
        self.class_13 = class_13
        synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_13id = {id: i for i, id in enumerate(synthia_set_13)}

        print("{} num images in Cityscapes {} set have been loaded.".format(
            len(self.items), self.split))

    def id2trainId(self, label, reverse=False, ignore_label=255):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        if self.class_16:
            label_copy_16 = ignore_label * \
                np.ones(label.shape, dtype=np.float32)
            for k, v in self.trainid_to_16id.items():
                label_copy_16[label_copy == k] = v
            label_copy = label_copy_16
        if self.class_13:
            label_copy_13 = ignore_label * \
                np.ones(label.shape, dtype=np.float32)
            for k, v in self.trainid_to_13id.items():
                label_copy_13[label_copy == k] = v
            label_copy = label_copy_13
        return label_copy

    def __getitem__(self, item):
        id = self.items[item]
        filename = id.split("train_")[-1].split("val_")[-1].split("test_")[-1]
        image_filepath = os.path.join(self.image_filepath,
                                      id.split("_")[0], id.split("_")[1])
        image_filename = filename + "_leftImg8bit.png"
        image_path = os.path.join(image_filepath, image_filename)
        image = Image.open(image_path).convert("RGB")

        gt_filepath = os.path.join(self.gt_filepath,
                                   id.split("_")[0], id.split("_")[1])
        gt_filename = filename + "_gtFine_labelIds.png"
        gt_image_path = os.path.join(gt_filepath, gt_filename)
        gt_image = Image.open(gt_image_path)

        if (self.split == "train" or
                self.split == "trainval") and self.training:
            image, gt_image, edge_mask = self._train_sync_transform(image,
                                                                    gt_image)
            return image, gt_image, edge_mask
        else:
            image, gt_image, edge_mask = self._val_sync_transform(image,
                                                                  gt_image)
            return image, gt_image, edge_mask, id

    def _train_sync_transform(self, img, mask):
        '''
        :param image:  PIL input image
        :param gt_image: PIL input gt_image
        :return:
        '''
        if self.random_mirror:
            # random mirror
            a = random.random()
            if a < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if mask:
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            crop_w, crop_h = self.crop_size

        if self.random_crop:
            # random scale
            base_w, base_h = self.base_size
            w, h = img.size
            assert w >= h
            if (base_w / w) > (base_h / h):
                base_size = base_w
                short_size = random.randint(
                    int(base_size * 0.5), int(base_size * 2.0))
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                base_size = base_h
                short_size = random.randint(
                    int(base_size * 0.5), int(base_size * 2.0))
                oh = short_size
                ow = int(1.0 * w * oh / h)

            img = img.resize((ow, oh), Image.BICUBIC)
            if mask:
                mask = mask.resize((ow, oh), Image.NEAREST)
            # pad crop
            if ow < crop_w or oh < crop_h:
                padh = crop_h - oh if oh < crop_h else 0
                padw = crop_w - ow if ow < crop_w else 0
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
                if mask:
                    mask = ImageOps.expand(
                        mask, border=(0, 0, padw, padh), fill=0)
            # random crop crop_size
            w, h = img.size
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            if mask:
                mask = mask.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        elif self.resize:
            img = img.resize(self.crop_size, Image.BICUBIC)
            if mask:
                mask = mask.resize(self.crop_size, Image.NEAREST)

        if self.gaussian_blur:
            # gaussian blur as in PSP
            b = random.random()
            c = random.random()
            # print(a,b,c)
            if b < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=c))
        # final transform
        if mask:
            img = self._img_transform(img)
            mask, edge_mask = self._mask_transform(mask)
            return img, mask, edge_mask
        else:
            img = self._img_transform(img)
            return img

    def _val_sync_transform(self, img, mask):
        if self.random_crop:
            crop_w, crop_h = self.crop_size
            w, h = img.size
            if crop_w / w < crop_h / h:
                oh = crop_h
                ow = int(1.0 * w * oh / h)
            else:
                ow = crop_w
                oh = int(1.0 * h * ow / w)
            img = img.resize((ow, oh), Image.BICUBIC)
            mask = mask.resize((ow, oh), Image.NEAREST)
            # center crop
            w, h = img.size
            x1 = int(round((w - crop_w) / 2.))
            y1 = int(round((h - crop_h) / 2.))
            img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            mask = mask.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        elif self.resize:
            img = img.resize(self.crop_size, Image.BICUBIC)
            mask = mask.resize(self.crop_size, Image.NEAREST)

        # final transform
        img = self._img_transform(img)
        mask, edge_mask = self._mask_transform(mask)
        return img, mask, edge_mask

    def _img_transform(self, image):
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)

        new_image = paddle.to_tensor(image)

        return new_image

    def _mask_transform(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        edge_mask = None
        if self.edge:
            edge_mask = F.mask_to_binary_edge(
                target, radius=1, num_classes=self.NUM_CLASSES)

        target = paddle.to_tensor(target)

        return target, edge_mask

    def __len__(self):
        return len(self.items)

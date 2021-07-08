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

from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
import paddleseg.transforms.functional as F
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance
import copy


def padding(img_ori, mask_ori, size=224, padding_color=128):
    height = img_ori.shape[0]
    width = img_ori.shape[1]

    img = np.zeros((max(height, width), max(height, width), 3)) + padding_color
    mask = np.zeros((max(height, width), max(height, width)))

    if (height > width):
        padding = int((height - width) / 2)
        img[:, padding:padding + width, :] = img_ori
        mask[:, padding:padding + width] = mask_ori
    else:
        padding = int((width - height) / 2)
        img[padding:padding + height, :, :] = img_ori
        mask[padding:padding + height, :] = mask_ori

    img = np.uint8(img)
    mask = np.uint8(mask)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_CUBIC)

    return np.array(img, dtype=np.float32), np.array(mask, dtype=np.float32)


def data_aug_blur(image, prob=0.5):
    if random.random() < prob:
        return image

    select = random.random()
    if select < 0.3:
        kernalsize = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernalsize, kernalsize), 0)
    elif select < 0.6:
        kernalsize = random.choice([3, 5])
        image = cv2.medianBlur(image, kernalsize)
    else:
        kernalsize = random.choice([3, 5])
        image = cv2.blur(image, (kernalsize, kernalsize))
    return image


def data_aug_color(image, prob=0.5):
    if random.random() < prob:
        return image
    random_factor = np.random.randint(4, 17) / 10.
    color_image = ImageEnhance.Color(image).enhance(random_factor)
    random_factor = np.random.randint(4, 17) / 10.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(
        random_factor)
    random_factor = np.random.randint(6, 15) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(
        random_factor)
    random_factor = np.random.randint(8, 13) / 10.
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)


def data_aug_noise(image, prob=0.5):
    if random.random() < prob:
        return image
    mu = 0
    sigma = random.random() * 10.0
    image = np.array(image, dtype=np.float32)
    image += np.random.normal(mu, sigma, image.shape)
    image[image > 255] = 255
    image[image < 0] = 0
    return image


def show_edge(mask_ori):
    mask = mask_ori.copy()
    # find countours: img must be binary
    myImg = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    ret, binary = cv2.threshold(
        np.uint8(mask) * 255, 127, 255, cv2.THRESH_BINARY)
    try:
        countours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # RETR_EXTERNAL
    except Exception:
        pass
    '''
    cv2.drawContours(myImg, countours, -1, 1, 10)
    diff = mask + myImg
    diff[diff < 2] = 0
    diff[diff == 2] = 1
    return diff
    '''
    cv2.drawContours(myImg, countours, -1, 1, 4)
    # myImg[myImg==1] = 255
    # cv2.imwrite('eage.jpg', myImg)
    return myImg


@manager.DATASETS.add_component
class EG1800(Dataset):
    """
    EG1800 dataset `http://xiaoyongshen.me/webpage_portrait/index.html`.

    Args:
        transforms (list): A list of image transformations.
        dataset_root (str, optional): The ADK20K dataset directory. Default: None.
        mode (str, optional): A subset of the entire dataset. It should be one of ('train', 'val'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 2

    def __init__(self, transforms, dataset_root=None, mode='train', edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.ignore_index = 255
        self.mode = mode
        self.num_classes = self.NUM_CLASSES
        self.input_width = 224
        self.input_height = 224
        self.std = [0.23, 0.23, 0.23]
        self.mean = [0.485, 0.458, 0.408]

        if mode == 'train':
            path = os.path.join(dataset_root, 'eg1800_train.txt')
        else:
            path = os.path.join(dataset_root, 'eg1800_test.txt')
        with open(path, 'r') as f:
            files = f.readlines()
        img_files = [
            os.path.join(dataset_root, 'Images', file).strip() for file in files
        ]
        label_files = [
            os.path.join(dataset_root, 'Labels', file).strip() for file in files
        ]

        self.file_list = [[
            img_path, label_path
        ] for img_path, label_path in zip(img_files, label_files)]
        pass

    def __getitem__(self, item):
        image_path, label_path = self.file_list[item]
        im = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)
        label[label > 1] = 0

        if self.mode == "val":
            im, label = padding(im, label)
            im, label = self.transforms(im=im, label=label)
            im = np.float32(im[::-1, :, :])
            im_aug = copy.deepcopy(im)
        else:
            im, label = self.transforms(im=im, label=label)
            # add augmentation
            std = np.array(self.std)[:, np.newaxis, np.newaxis]
            mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
            im_aug = im * std + mean
            im_aug *= 255.0

            im_aug = np.transpose(im_aug, [1, 2, 0]).astype('uint8')
            im_aug = Image.fromarray(im_aug)
            im_aug = data_aug_color(im_aug)
            im_aug = np.asarray(im_aug)
            im_aug = data_aug_blur(im_aug)
            im_aug = data_aug_noise(im_aug)
            im_aug = np.transpose(im_aug, [2, 0, 1])
            im_aug = F.normalize(im_aug, mean, std)
            im_aug = np.float32(im_aug[::-1, :, :])
            im = np.float32(im[::-1, :, :])

        label = cv2.resize(np.uint8(label),
                           (self.input_width, self.input_height),
                           interpolation=cv2.INTER_NEAREST)

        # add mask blur
        label = np.uint8(cv2.blur(label, (5, 5)))
        label[label >= 0.5] = 1
        label[label < 0.5] = 0

        edge = show_edge(label)
        im = np.concatenate([im_aug, im])
        if self.mode == "train":
            return im, label, edge
        else:
            return im, label

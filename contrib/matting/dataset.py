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

import cv2
import numpy as np
import random
import paddle

from utils import get_files
import transforms as T


def gen_trimap(alpha):
    k_size = random.choice(range(2, 5))
    iterations = np.random.randint(5, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(alpha, kernel, iterations=iterations)
    eroded = cv2.erode(alpha, kernel, iterations=iterations)
    trimap = np.zeros(alpha.shape)
    trimap.fill(128)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0

    return trimap


class Dataset(paddle.io.Dataset):
    def __init__(self):
        self.png_file = '/mnt/chenguowei01/datasets/matting/PhotoMatte85/0051115Q_000001_0041.png'
        self.background = np.zeros((3, 320, 320), dtype='float32')
        self.background[1, :, :] = 255

    def __getitem__(self, idx):
        img_png = cv2.imread(self.png_file, cv2.IMREAD_UNCHANGED)
        img_png = cv2.resize(img_png, (320, 320))
        img_png = np.transpose(img_png, [2, 0, 1])
        alpha = img_png[-1, :, :].astype('float32')

        img = img_png[:-1, :, :].astype('float32')
        img = img[::-1, :, :]
        img = (img / 255 - 0.5) / 0.5
        # img = (alpha/255) * img + (1-alpha/255) * self.background

        trimap = gen_trimap(alpha).astype('float32')

        return img, alpha, trimap

    def __len__(self):
        return 1000


class HumanDataset(paddle.io.Dataset):
    def __init__(
            self,
            dataset_root,
            transforms,
            mode='train',
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.transforms = T.Compose(transforms)

        img_dir = os.path.join(dataset_root, mode, 'image')
        self.img_list = get_files(img_dir)  # a list
        self.alpha_list = [f.replace('image', 'alpha') for f in self.img_list]
        self.fg_list = [f.replace('image', 'fg') for f in self.img_list]
        self.bg_list = [f.replace('image', 'bg') for f in self.img_list]

    def __getitem__(self, idx):
        data = {}
        data['img'] = self.img_list[idx]
        data['alpha'] = self.alpha_list[idx]
        data['fg'] = self.fg_list[idx]
        data['bg'] = self.bg_list[idx]
        data['gt_fields'] = ['alpha', 'fg', 'bg']

        data = self.transforms(data)
        data['img'] = data['img'].astype('float32')
        for key in data.get('gt_fields', []):
            data[key] = data[key].astype('float32')
        data['trimap'] = gen_trimap(data['alpha']).astype('float32')

        return data

    def __len__(self):
        return len(self.img_list)

    @property
    def gen_trimap(alpha):
        k_size = random.choice(range(2, 5))
        iterations = np.random.randint(5, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(alpha, kernel, iterations=iterations)
        eroded = cv2.erode(alpha, kernel, iterations=iterations)
        trimap = np.zeros(alpha.shape)
        trimap.fill(128)
        trimap[eroded >= 255] = 255
        trimap[dilated <= 0] = 0

        return trimap


if __name__ == '__main__':
    t = [T.LoadImages(), T.Resize(), T.Normalize()]
    train_dataset = HumanDataset(
        dataset_root='/mnt/chenguowei01/datasets/matting/human_matting/',
        transforms=t,
        mode='val')
    print(train_dataset.img_list[0], len(train_dataset.img_list),
          len(train_dataset.alpha_list), len(train_dataset.fg_list),
          len(train_dataset.bg_list))
    data = train_dataset[0]
    print(np.min(data['img']), np.max(data['img']))
    print(data['img'].shape, data['fg'].shape, data['bg'].shape,
          data['alpha'].shape, data['trimap'].shape)

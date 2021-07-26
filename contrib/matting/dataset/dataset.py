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


class Dataset(paddle.io.Dataset):
    """
    The dataset folder should be as follow:
    root
    |__train
    |  |__image
    |  |__fg
    |  |__bg
    |  |__alpha
    |
    |__val
    |  |__image
    |  |__fg
    |  |__bg
    |  |__alpha
    |  |__[trimap]

    """

    def __init__(
            self,
            dataset_root,
            transforms,
            mode='train',
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.transforms = T.Compose(transforms)
        self.mode = mode

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
        data['gt_field'] = []

        if self.mode == 'train':
            data['gt_fields'] = ['alpha', 'fg', 'bg']
        else:
            data['gt_fields'] = ['alpha']
            data['img_name'] = self.img_list[idx].lstrip(
                self.dataset_root)  # using in save prediction results
            # If has trimap, use it
            trimap_path = data['alpha'].replace('alpha', 'trimap')
            if os.path.exists(trimap_path):
                data['trimap'] = trimap_path
                data['gt_fields'].append('trimap')

        data['trans_info'] = []  # Record shape change information
        data = self.transforms(data)
        data['img'] = data['img'].astype('float32')
        for key in data.get('gt_fields', []):
            data[key] = data[key].astype('float32')
        if 'trimap' not in data:
            data['trimap'] = self.gen_trimap(
                data['alpha'], mode=self.mode).astype('float32')

        return data

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def gen_trimap(alpha, mode='train', eval_kernel=7):
        if mode == 'train':
            k_size = random.choice(range(2, 5))
            iterations = np.random.randint(5, 15)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (k_size, k_size))
            dilated = cv2.dilate(alpha, kernel, iterations=iterations)
            eroded = cv2.erode(alpha, kernel, iterations=iterations)
            trimap = np.zeros(alpha.shape)
            trimap.fill(128)
            trimap[eroded > 254.5] = 255
            trimap[dilated < 0.5] = 0
        else:
            k_size = eval_kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (k_size, k_size))
            dilated = cv2.dilate(alpha, kernel)
            trimap = np.zeros(alpha.shape)
            trimap.fill(128)
            trimap[alpha >= 250] = 255
            trimap[dilated <= 5] = 0

        return trimap


if __name__ == '__main__':
    t = [T.LoadImages(), T.Resize(), T.Normalize()]
    train_dataset = Dataset(
        dataset_root='data/matting/human_matte/', transforms=t, mode='train')
    print(len(train_dataset))

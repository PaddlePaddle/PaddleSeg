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
import math

import cv2
import numpy as np
import random
import paddle

import transforms as T


class HumanMattingDataset(paddle.io.Dataset):
    """
    human_matting
    |__Composition-1k（origin dataset name）
    |    |__train
    |    |    |__fg
    |    |    |__alpha
    |    |__val
    |         |__fg
    |         |__alpha
    |         |__trimap
    |__Distinctions-646
    |
    |__bg (background）
    |    |__coco_17
    |    |__pascal_voc12
    |
    |__train.txt
    |__val.tat


    Args:
        dataset_root(str): The root path of dataset
        transforms(list):  Transforms for image.
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'trainval'). Default: 'train'.
        train_file (str|list, optional): File list is used to train. It should be `foreground_image.png background_image.png`
            or `foreground_image.png`. It shold be provided if mode equal to 'train'. Default: None.
        val_file (str|list, optional): File list is used to evaluation. It should be `foreground_image.png background_image.png`
            or `foreground_image.png`. It shold be provided if mode equal to 'val'. Default: None.

    """

    def __init__(self,
                 dataset_root,
                 transforms,
                 mode='train',
                 train_file=None,
                 val_file=None,
                 get_trimap=True):
        super().__init__()
        self.dataset_root = dataset_root
        self.transforms = T.Compose(transforms)
        self.mode = mode
        self.get_trimap = get_trimap

        # check file
        if mode == 'train' or mode == 'trainval':
            if train_file is None:
                raise ValueError(
                    "When `mode` is 'train', `train_file must be provided!")
            if isinstance(train_file, str):
                train_file = [train_file]
            file_list = train_file

        if mode == 'val' or mode == 'trainval':
            if val_file is None:
                raise ValueError(
                    "When `mode` is 'val', `val_file must be provided!")
            if isinstance(val_file, str):
                val_file = [val_file]
            file_list = val_file

        if mode == 'trainval':
            file_list = train_file + val_file

        # read file
        self.fg_bg_list = []
        for file in file_list:
            file = os.path.join(dataset_root, file)
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    self.fg_bg_list.append(line)

    def __getitem__(self, idx):
        data = {}
        fg_bg_file = self.fg_bg_list[idx]
        fg_bg_file = fg_bg_file.split(' ')
        data['img_name'] = fg_bg_file[0]  # using in save prediction results
        fg_file = os.path.join(self.dataset_root, fg_bg_file[0])
        alpha_file = fg_file.replace('fg', 'alpha')
        fg = cv2.imread(fg_file)
        alpha = cv2.imread(alpha_file, 0)
        data['alpha'] = alpha
        data['gt_fields'] = []

        if len(fg_bg_file) == 2:
            bg_file = os.path.join(self.dataset_root, fg_bg_file[1])
            bg = cv2.imread(bg_file)
            data['img'], data['bg'] = self.composite(fg, alpha, bg)
            data['fg'] = fg
            if self.mode in ['train', 'trainval']:
                data['gt_fields'].append('fg')
                data['gt_fields'].append('bg')
                data['gt_fields'].append('alpha')
        else:
            data['img'] = fg
            if self.mode in ['train', 'trainval']:
                data['fg'] = fg.copy()
                data['bg'] = fg.copy()
                data['gt_fields'].append('fg')
                data['gt_fields'].append('bg')
                data['gt_fields'].append('alpha')

        data['trans_info'] = []  # Record shape change information
        data = self.transforms(data)

        # When evaluation, gt should not be transforms.
        if self.mode == 'val':
            data['gt_fields'].append('alpha')

        data['img'] = data['img'].astype('float32')
        for key in data.get('gt_fields', []):
            data[key] = data[key].astype('float32')
        if self.get_trimap:
            # Trimap read from file only happening in evaluation.
            if self.mode == 'val':
                trimap_path = alpha_file.replace('alpha', 'trimap')
                if os.path.exists(trimap_path):
                    data['trimap'] = trimap_path
                    data['gt_fields'].append('trimap')

            if 'trimap' not in data:
                data['trimap'] = self.gen_trimap(
                    data['alpha'], mode=self.mode).astype('float32')
            data['trimap'] = data['trimap'][np.newaxis, :, :]

        data['alpha'] = data['alpha'][np.newaxis, :, :] / 255.

        return data

    def __len__(self):
        return len(self.fg_bg_list)

    def composite(self, fg, alpha, ori_bg):
        fg_h, fg_w = fg.shape[:2]
        ori_bg_h, ori_bg_w = ori_bg.shape[:2]

        wratio = fg_w / ori_bg_w
        hratio = fg_h / ori_bg_h
        ratio = wratio if wratio > hratio else hratio

        # Resize ori_bg if it is smaller than fg.
        if ratio > 1:
            resize_h = math.ceil(ori_bg_h * ratio)
            resize_w = math.ceil(ori_bg_w * ratio)
            bg = cv2.resize(
                ori_bg, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        else:
            bg = ori_bg

        bg = bg[0:fg_h, 0:fg_w, :]
        alpha = alpha / 255
        alpha = np.expand_dims(alpha, axis=2)
        image = alpha * fg + (1 - alpha) * bg
        image = image.astype(np.uint8)
        return image, bg

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
    t = [T.LoadImages(to_rgb=False), T.Resize(), T.Normalize()]
    train_dataset = HumanMattingDataset(
        dataset_root='../data/matting/human_matte/',
        transforms=t,
        mode='val',
        train_file=['Composition-1k_train.txt', 'Distinctions-646_train.txt'],
        val_file=['Composition-1k_val.txt', 'Distinctions-646_val.txt'])
    data = train_dataset[81]
    print(data.keys())
    print(data['gt_fields'])

    data['img'] = np.transpose(data['img'], (1, 2, 0))
    for key in data.get('gt_fields', []):
        if len(data[key].shape) == 2:
            continue
        data[key] = np.transpose(data[key], (1, 2, 0))

    data['img'] = ((data['img'] * 0.5 + 0.5) * 255).astype('uint8')
    for key in data['gt_fields']:
        if key == 'alpha':
            continue
        data[key] = ((data[key] * 0.5 + 0.5) * 255).astype('uint8')

    cv2.imwrite('img.png', data['img'])
    for key in data['gt_fields']:
        cv2.imwrite(key + '.png', data[key])

    cv2.imwrite('trimap.png', data['trimap'].astype('uint8'))

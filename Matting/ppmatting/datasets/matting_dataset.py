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
from paddleseg.cvlibs import manager

import ppmatting.transforms as T


@manager.DATASETS.add_component
class MattingDataset(paddle.io.Dataset):
    """
    Pass in a dataset that conforms to the format.
        matting_dataset/
        |--bg/
        |
        |--train/
        |  |--fg/
        |  |--alpha/
        |
        |--val/
        |  |--fg/
        |  |--alpha/
        |  |--trimap/ (if existing)
        |
        |--train.txt
        |
        |--val.txt
    See README.md for more information of dataset.

    Args:
        dataset_root(str): The root path of dataset.
        transforms(list):  Transforms for image.
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'trainval'). Default: 'train'.
        train_file (str|list, optional): File list is used to train. It should be `foreground_image.png background_image.png`
            or `foreground_image.png`. It shold be provided if mode equal to 'train'. Default: None.
        val_file (str|list, optional): File list is used to evaluation. It should be `foreground_image.png background_image.png`
            or `foreground_image.png` or ``foreground_image.png background_image.png trimap_image.png`.
            It shold be provided if mode equal to 'val'. Default: None.
        get_trimap (bool, optional): Whether to get triamp. Default: True.
        separator (str, optional): The separator of train_file or val_file. If file name contains ' ', '|' may be perfect. Default: ' '.
        key_del (tuple|list, optional): The key which is not need will be delete to accellect data reader. Default: None.
        if_rssn (bool, optional): Whether to use RSSN while Compositing image. Including denoise and blur. Default: False.
    """

    def __init__(self,
                 dataset_root,
                 transforms,
                 mode='train',
                 train_file=None,
                 val_file=None,
                 get_trimap=True,
                 separator=' ',
                 key_del=None,
                 if_rssn=False):
        super().__init__()
        self.dataset_root = dataset_root
        self.transforms = T.Compose(transforms)
        self.mode = mode
        self.get_trimap = get_trimap
        self.separator = separator
        self.key_del = key_del
        self.if_rssn = if_rssn

        # check file
        if mode == 'train' or mode == 'trainval':
            if train_file is None:
                raise ValueError(
                    "When `mode` is 'train' or 'trainval', `train_file must be provided!"
                )
            if isinstance(train_file, str):
                train_file = [train_file]
            file_list = train_file

        if mode == 'val' or mode == 'trainval':
            if val_file is None:
                raise ValueError(
                    "When `mode` is 'val' or 'trainval', `val_file must be provided!"
                )
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
        if mode != 'val':
            random.shuffle(self.fg_bg_list)

    def __getitem__(self, idx):
        data = {}
        fg_bg_file = self.fg_bg_list[idx]
        fg_bg_file = fg_bg_file.split(self.separator)
        data['img_name'] = fg_bg_file[0]  # using in save prediction results
        fg_file = os.path.join(self.dataset_root, fg_bg_file[0])
        alpha_file = fg_file.replace('/fg', '/alpha')
        fg = cv2.imread(fg_file)
        alpha = cv2.imread(alpha_file, 0)
        data['alpha'] = alpha
        data['gt_fields'] = []

        # line is: fg [bg] [trimap]
        if len(fg_bg_file) >= 2:
            bg_file = os.path.join(self.dataset_root, fg_bg_file[1])
            bg = cv2.imread(bg_file)
            data['img'], data['fg'], data['bg'] = self.composite(fg, alpha, bg)
            if self.mode in ['train', 'trainval']:
                data['gt_fields'].append('fg')
                data['gt_fields'].append('bg')
                data['gt_fields'].append('alpha')
            if len(fg_bg_file) == 3 and self.get_trimap:
                if self.mode == 'val':
                    trimap_path = os.path.join(self.dataset_root, fg_bg_file[2])
                    if os.path.exists(trimap_path):
                        data['trimap'] = trimap_path
                        data['gt_fields'].append('trimap')
                        data['ori_trimap'] = cv2.imread(trimap_path, 0)
                    else:
                        raise FileNotFoundError(
                            'trimap is not Found: {}'.format(fg_bg_file[2]))
        else:
            data['img'] = fg
            if self.mode in ['train', 'trainval']:
                data['fg'] = fg.copy()
                data['bg'] = fg.copy()
                data['gt_fields'].append('fg')
                data['gt_fields'].append('bg')
                data['gt_fields'].append('alpha')

        data['trans_info'] = []  # Record shape change information

        # Generate trimap from alpha if no trimap file provided
        if self.get_trimap:
            if 'trimap' not in data:
                data['trimap'] = self.gen_trimap(
                    data['alpha'], mode=self.mode).astype('float32')
                data['gt_fields'].append('trimap')
                if self.mode == 'val':
                    data['ori_trimap'] = data['trimap'].copy()

        # Delete key which is not need
        if self.key_del is not None:
            for key in self.key_del:
                if key in data.keys():
                    data.pop(key)
                if key in data['gt_fields']:
                    data['gt_fields'].remove(key)
        data = self.transforms(data)

        # When evaluation, gt should not be transforms.
        if self.mode == 'val':
            data['gt_fields'].append('alpha')

        data['img'] = data['img'].astype('float32')
        for key in data.get('gt_fields', []):
            data[key] = data[key].astype('float32')

        if 'trimap' in data:
            data['trimap'] = data['trimap'][np.newaxis, :, :]
        if 'ori_trimap' in data:
            data['ori_trimap'] = data['ori_trimap'][np.newaxis, :, :]

        data['alpha'] = data['alpha'][np.newaxis, :, :] / 255.

        return data

    def __len__(self):
        return len(self.fg_bg_list)

    def composite(self, fg, alpha, ori_bg):
        if self.if_rssn:
            if np.random.rand() < 0.5:
                fg = cv2.fastNlMeansDenoisingColored(fg, None, 3, 3, 7, 21)
                ori_bg = cv2.fastNlMeansDenoisingColored(ori_bg, None, 3, 3, 7,
                                                         21)
            if np.random.rand() < 0.5:
                radius = np.random.choice([19, 29, 39, 49, 59])
                ori_bg = cv2.GaussianBlur(ori_bg, (radius, radius), 0, 0)
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
        return image, fg, bg

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

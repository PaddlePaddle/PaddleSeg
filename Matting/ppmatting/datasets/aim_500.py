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
import random
import numpy as np
from PIL import Image

import paddle
from paddleseg.cvlibs import manager

import ppmatting.transforms as T


@manager.DATASETS.add_component
class AIM500(paddle.io.Dataset):
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
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # check file (only for test)
        if mode == 'val':
            if val_file is None:
                raise ValueError(
                    "When `mode` is 'val' or 'trainval', `val_file must be provided!"
                )
            if isinstance(val_file, str):
                val_file = [val_file]
            file_list = val_file

        # read file
        self.fg_list = []
        for file in file_list:
            file = os.path.join(dataset_root, file)
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    self.fg_list.append(line)
        if mode != 'val':
            random.shuffle(self.fg_list)

    def __getitem__(self, idx):
        data = {}
        data['trans_info'] = []  # Record shape change information

        # load images
        fg_file = self.fg_list[idx]
        data['img_name'] = fg_file.split("/")[-1]  # using in save prediction results
        fg_file = os.path.join(self.dataset_root, fg_file)
        alpha_file = fg_file.replace('/fg', '/alpha').replace('.jpg', '.png')
        fg = np.array(Image.open(fg_file))
        alpha = np.array(Image.open(alpha_file))
        data['img'] = fg[:, :, :3] if fg.ndim > 2 else fg
        data['alpha'] = alpha[:, :, 0] if alpha.ndim > 2 else alpha
        data['gt_fields'] = []

        if self.get_trimap:
            trimap_file = fg_file.replace('/fg', '/trimap').replace('.jpg', '.png')
            if os.path.exists(trimap_file):
                trimap = np.array(Image.open(trimap_file))
                trimap = trimap[:, :, 0] if trimap.ndim > 2 else trimap
                data['trimap'] = trimap
                data['gt_fields'].append('trimap')
                if self.mode == 'val':
                    data['ori_trimap'] = data['trimap'].copy()
            else:
                raise FileNotFoundError('trimap is not Found: {}'.format(trimap_file))
        
        data = self.transforms(data)
        normalize = paddle.vision.transforms.Normalize(mean=self.mean, std=self.std)
        
        if data.get('img_g') is not None:
            data['img_g'] = paddle.to_tensor(data['img_g']).transpose(perm=[2, 0, 1]) / 255.0
            data['img_g'] = normalize(data['img_g'])

        if data.get('img_l') is not None:
            data['img_l'] = paddle.to_tensor(data['img_l']).transpose(perm=[2, 0, 1]) / 255.0
            data['img_l'] = normalize(data['img_l'])

        # Delete key which is not need
        if self.key_del is not None:
            for key in self.key_del:
                if key in data.keys():
                    data.pop(key)
                if key in data['gt_fields']:
                    data['gt_fields'].remove(key)

        # When evaluation, gt should not be transforms.
        data['img'] = data['img'].astype('float32')
        for key in data.get('gt_fields', []):
            data[key] = data[key].astype('float32')

        if self.mode == 'val':
            data['gt_fields'].append('alpha')
        
        if 'trimap' in data:
            data['trimap'] = data['trimap'][np.newaxis, :, :]
        if 'ori_trimap' in data:
            data['ori_trimap'] = data['ori_trimap'][np.newaxis, :, :]

        data['alpha'] = data['alpha'][np.newaxis, :, :] / 255.

        return data

    def __len__(self):
        return len(self.fg_list)

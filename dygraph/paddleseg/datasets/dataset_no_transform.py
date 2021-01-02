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
import glob

import paddle
import numpy as np
from PIL import Image

from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class DatasetNoTransform(paddle.io.Dataset):
    """
    Pass in a custom dataset that conforms to the format.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory.
        num_classes (int): Number of classes.
        mode (str): which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        train_path (str): The train dataset file. When mode is 'train', train_path is necessary.
            The contents of train_path file are as follow:
            image1.jpg ground_truth1.png
            image2.jpg ground_truth2.png
        val_path (str): The evaluation dataset file. When mode is 'val', val_path is necessary.
            The contents is the same as train_path
        test_path (str): The test dataset file. When mode is 'test', test_path is necessary.
            The annotation file is not necessary in test_path file.
        separator (str): The separator of dataset list. Default: ' '.

        Examples:

            import paddleseg.transforms as T
            from paddleseg.datasets import Dataset

            transforms = [T.RandomPaddingCrop(crop_size=(512,512)), T.Normalize()]
            dataset_root = 'dataset_root_path'
            train_path = 'train_path'
            num_classes = 2
            dataset = Dataset(transforms = transforms,
                              dataset_root = dataset_root,
                              num_classes = 2,
                              train_path = train_path,
                              mode = 'train')

    """

    def __init__(self, img_dir, label_dir, mode='val'):
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        self.num_classes = 19
        self.ignore_index = 255

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
            raise FileNotFoundError(
                'Do not find images directory {} or labels directory {}'.format(
                    img_dir, label_dir))

        label_files = sorted(
            glob.glob(
                os.path.join(label_dir, '*', '*_gtFine_labelTrainIds.png')))
        # img_files = sorted(
        #     glob.glob(os.path.join(img_dir, '*',
        #                      '*_gtFine_labelTrainIds.png')))
        img_files = sorted(
            glob.glob(os.path.join(img_dir, '*_leftImg8bit.png')))

        if len(img_files) != len(label_files):
            raise ValueError(
                "The number of images = {} is not equal to the number of labels = {} in Cityscapes dataset."
                .format(len(img_files), len(label_files)))

        self.file_list = [[
            img_path, label_path
        ] for img_path, label_path in zip(img_files, label_files)]

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        if self.mode == 'test':
            im, _ = read_img(image_path)
            im = im[np.newaxis, ...]
            return im, image_path
        elif self.mode == 'val':
            im, _ = read_img(image_path)
            label = np.asarray(Image.open(label_path))
            label = label[np.newaxis, :, :]
            return im, label
        else:
            im, label = read_img(im=image_path, label=label_path)
            return im, label

    def __len__(self):
        return len(self.file_list)


def read_img(im, label=None, to_rgb=True):
    if isinstance(im, str):
        im = np.asarray(Image.open(im))
    if isinstance(label, str):
        label = np.asarray(Image.open(label))
    if im is None:
        raise ValueError('Can\'t read The image file {}!'.format(im))
    return im, label

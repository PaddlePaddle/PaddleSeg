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

import paddleseg.env as segenv
from .dataset import Dataset
from paddleseg.utils.download import download_file_and_uncompress
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose

URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"


@manager.DATASETS.add_component
class PascalVOC(Dataset):
    """Pascal VOC dataset `http://host.robots.ox.ac.uk/pascal/VOC/`. If you want to augment the dataset,
    please run the voc_augment.py in tools.
    Args:
        dataset_root: The dataset directory.
        mode: Which part of dataset to use.. it is one of ('train', 'val', 'test'). Default: 'train'.
        transforms: Transforms for image.
        download: Whether to download dataset if dataset_root is None.
    """

    def __init__(self,
                 dataset_root=None,
                 mode='train',
                 transforms=None,
                 download=True):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.mode = mode
        self.file_list = list()
        self.num_classes = 21
        self.ignore_index = 255

        if mode.lower() not in ['train', 'trainval', 'trainaug', 'val']:
            raise Exception(
                "`mode` should be one of ('train', 'trainval', 'trainaug', 'val') in PascalVOC dataset, but got {}."
                .format(mode))

        if self.transforms is None:
            raise Exception("`transforms` is necessary, but it is None.")

        if self.dataset_root is None:
            if not download:
                raise Exception(
                    "`dataset_root` not set and auto download disabled.")
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=segenv.DATA_HOME,
                extrapath=segenv.DATA_HOME,
                extraname='VOCdevkit')
        elif not os.path.exists(self.dataset_root):
            raise Exception('there is not `dataset_root`: {}.'.format(
                self.dataset_root))

        image_set_dir = os.path.join(self.dataset_root, 'VOC2012', 'ImageSets',
                                     'Segmentation')
        if mode == 'train':
            file_list = os.path.join(image_set_dir, 'train.txt')
        elif mode == 'val':
            file_list = os.path.join(image_set_dir, 'val.txt')
        elif mode == 'trainval':
            file_list = os.path.join(image_set_dir, 'trainval.txt')
        elif mode == 'trainaug':
            file_list = os.path.join(image_set_dir, 'train.txt')
            file_list_aug = os.path.join(image_set_dir, 'aug.txt')

            if not os.path.exists(file_list_aug):
                raise Exception(
                    "When `mode` is 'trainaug', Pascal Voc dataset should be augmented, "
                    "Please make sure voc_augment.py has been properly run when using this mode."
                )

        img_dir = os.path.join(self.dataset_root, 'VOC2012', 'JPEGImages')
        grt_dir = os.path.join(self.dataset_root, 'VOC2012',
                               'SegmentationClass')
        grt_dir_aug = os.path.join(self.dataset_root, 'VOC2012',
                                   'SegmentationClassAug')

        with open(file_list, 'r') as f:
            for line in f:
                line = line.strip()
                image_path = os.path.join(img_dir, ''.join([line, '.jpg']))
                grt_path = os.path.join(grt_dir, ''.join([line, '.png']))
                self.file_list.append([image_path, grt_path])
        if mode == 'trainaug':
            with open(file_list_aug, 'r') as f:
                for line in f:
                    line = line.strip()
                    image_path = os.path.join(img_dir, ''.join([line, '.jpg']))
                    grt_path = os.path.join(grt_dir_aug, ''.join([line,
                                                                  '.png']))
                    self.file_list.append([image_path, grt_path])

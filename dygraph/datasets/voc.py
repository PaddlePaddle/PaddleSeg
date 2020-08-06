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
from .dataset import Dataset
from utils.download import download_file_and_uncompress

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset')
URL = "https://paddleseg.bj.bcebos.com/dataset/VOCtrainval_11-May-2012.tar"


class PascalVOC(Dataset):
    """Pascal VOC dataset `http://host.robots.ox.ac.uk/pascal/VOC/`. If you want to augment the dataset,
    please run the voc_augment.py in tools.
    Args:
        data_dir: The dataset directory.
        image_set: Which part of dataset to use. Generally, image_set is of ('train', 'val', 'trainval', 'trainaug). Default: 'train'.
        mode: Dataset usage. it is one of ('train', 'eva', 'test'). Default: 'train'.
        transforms: Transforms for image.
        download: Whether to download dataset if data_dir is None.
    """

    def __init__(self,
                 data_dir=None,
                 image_set='train',
                 mode='train',
                 transforms=None,
                 download=True):
        self.data_dir = data_dir
        self.transforms = transforms
        self.mode = mode
        self.file_list = list()
        self.num_classes = 21

        if image_set.lower() not in ['train', 'val', 'trainval', 'trainaug']:
            raise Exception(
                "image_set should be one of ('train', 'val', 'trainval', 'trainaug'), but got {}."
                .format(image_set))

        if mode.lower() not in ['train', 'eval', 'test']:
            raise Exception(
                "mode should be 'train', 'eval' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise Exception("transforms is necessary, but it is None.")

        if self.data_dir is None:
            if not download:
                raise Exception("data_file not set and auto download disabled.")
            self.data_dir = download_file_and_uncompress(
                url=URL,
                savepath=DATA_HOME,
                extrapath=DATA_HOME,
                extraname='VOCdevkit')
            print(self.data_dir)

        image_set_dir = os.path.join(self.data_dir, 'VOC2012', 'ImageSets',
                                     'Segmentation')
        if image_set == 'train':
            file_list = os.path.join(image_set_dir, 'train.txt')
        elif image_set == 'val':
            file_list = os.path.join(image_set_dir, 'val.txt')
        elif image_set == 'trainval':
            file_list = os.path.join(image_set_dir, 'trainval.txt')
        elif image_set == 'trainaug':
            file_list = os.path.join(image_set_dir, 'train.txt')
            file_list_aug = os.path.join(image_set_dir, 'aug.txt')

            if not os.path.exists(file_list_aug):
                raise Exception(
                    "When image_set is 'trainaug', Pascal Voc dataset should be augmented, "
                    "Please make sure voc_augment.py has been properly run when using this mode."
                )

        img_dir = os.path.join(self.data_dir, 'VOC2012', 'JPEGImages')
        grt_dir = os.path.join(self.data_dir, 'VOC2012', 'SegmentationClass')
        grt_dir_aug = os.path.join(self.data_dir, 'VOC2012',
                                   'SegmentationClassAug')

        with open(file_list, 'r') as f:
            for line in f:
                line = line.strip()
                image_path = os.path.join(img_dir, ''.join([line, '.jpg']))
                grt_path = os.path.join(grt_dir, ''.join([line, '.png']))
                self.file_list.append([image_path, grt_path])
        if image_set == 'trainaug':
            with open(file_list_aug, 'r') as f:
                for line in f:
                    line = line.strip()
                    image_path = os.path.join(img_dir, ''.join([line, '.jpg']))
                    grt_path = os.path.join(grt_dir_aug, ''.join([line,
                                                                  '.png']))
                    self.file_list.append([image_path, grt_path])

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
URL = "https://paddleseg.bj.bcebos.com/dataset/optic_disc_seg.zip"


class OpticDiscSeg(Dataset):
    def __init__(self,
                 dataset_root=None,
                 transforms=None,
                 mode='train',
                 download=True):
        self.dataset_root = dataset_root
        self.transforms = transforms
        self.file_list = list()
        self.mode = mode
        self.num_classes = 2

        if mode.lower() not in ['train', 'val', 'test']:
            raise Exception(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise Exception("transforms is necessary, but it is None.")

        if self.dataset_root is None:
            if not download:
                raise Exception("data_file not set and auto download disabled.")
            self.dataset_root = download_file_and_uncompress(
                url=URL, savepath=DATA_HOME, extrapath=DATA_HOME)

        if mode == 'train':
            file_list = os.path.join(self.dataset_root, 'train_list.txt')
        elif mode == 'val':
            file_list = os.path.join(self.dataset_root, 'val_list.txt')
        else:
            file_list = os.path.join(self.dataset_root, 'test_list.txt')

        with open(file_list, 'r') as f:
            for line in f:
                items = line.strip().split()
                if len(items) != 2:
                    if mode == 'train' or mode == 'val':
                        raise Exception(
                            "File list format incorrect! It should be"
                            " image_name label_name\\n")
                    image_path = os.path.join(self.dataset_root, items[0])
                    grt_path = None
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    grt_path = os.path.join(self.dataset_root, items[1])
                self.file_list.append([image_path, grt_path])

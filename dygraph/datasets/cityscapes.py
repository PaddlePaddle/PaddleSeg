#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.io import Dataset

from utils.download import download_file_and_uncompress

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset')
URL = "https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar"


class Cityscapes(Dataset):
    def __init__(self,
                 data_dir=None,
                 transforms=None,
                 mode='train',
                 download=True):
        self.data_dir = data_dir
        self.transforms = transforms
        self.file_list = list()
        self.mode = mode
        self.num_classes = 19

        if mode.lower() not in ['train', 'eval', 'test']:
            raise Exception(
                "mode should be 'train', 'eval' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise Exception("transform is necessary, but it is None.")

        self.data_dir = data_dir
        if self.data_dir is None:
            if not download:
                raise Exception("data_file not set and auto download disabled.")
            self.data_dir = download_file_and_uncompress(
                url=URL, savepath=DATA_HOME, extrapath=DATA_HOME)

        if mode == 'train':
            file_list = os.path.join(self.data_dir, 'train.list')
        elif mode == 'eval':
            file_list = os.path.join(self.data_dir, 'val.list')
        else:
            file_list = os.path.join(self.data_dir, 'test.list')

        with open(file_list, 'r') as f:
            for line in f:
                items = line.strip().split()
                if len(items) != 2:
                    if mode == 'train' or mode == 'eval':
                        raise Exception(
                            "File list format incorrect! It should be"
                            " image_name label_name\\n")
                    image_path = os.path.join(self.data_dir, items[0])
                    grt_path = None
                else:
                    image_path = os.path.join(self.data_dir, items[0])
                    grt_path = os.path.join(self.data_dir, items[1])
                self.file_list.append([image_path, grt_path])

    def __getitem__(self, idx):
        image_path, grt_path = self.file_list[idx]
        im, im_info, label = self.transforms(im=image_path, label=grt_path)
        if self.mode == 'train':
            return im, label
        elif self.mode == 'eval':
            return im, label
        if self.mode == 'test':
            return im, im_info, image_path

    def __len__(self):
        return len(self.file_list)

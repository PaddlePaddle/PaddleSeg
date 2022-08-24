# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import numpy as np

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))

from medicalseg.cvlibs import manager
from medicalseg.transforms import Compose
import paddle

URL = ' '  # todo: add coronavirus url


@manager.DATASETS.add_component
class Synapse(paddle.io.Dataset):
    def __init__(self,
                 dataset_root=None,
                 result_dir=None,
                 transforms=None,
                 num_classes=None,
                 mode='train',
                 dataset_json_path=""):
        super(Synapse, self).__init__()
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms, igmax=True)
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.dataset_json_path = dataset_json_path
        if self.mode == 'train':
            self.filenames = os.listdir(
                os.path.join(dataset_root, "train", 'img'))
        elif self.mode == 'val':
            self.filenames = os.listdir(
                os.path.join(dataset_root, "val", 'img'))
        else:
            raise ValueError(
                "`mode` should be 'train' or 'val', but got {}.".format(mode))

    def __getitem__(self, idx):
        if self.mode == "train":
            image_path = os.path.join(self.dataset_root, 'train', 'img',
                                      self.filenames[idx])
            label_path = os.path.join(self.dataset_root, 'train', 'label',
                                      self.filenames[idx].replace("img",
                                                                  'label'))
        else:
            image_path = os.path.join(self.dataset_root, 'val', 'img',
                                      self.filenames[idx])
            label_path = os.path.join(self.dataset_root, 'val', 'label',
                                      self.filenames[idx].replace("img",
                                                                  'label'))

        im, label = self.transforms(im=image_path, label=label_path)

        return im.astype('float32'), label, self.filenames[idx]  # npy file name

    def __len__(self):
        return len(self.filenames)

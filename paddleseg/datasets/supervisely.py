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

from paddleseg.datasets import EG1800
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class SUPERVISELY(EG1800):
    """
    Supervise.ly dataset `https://supervise.ly/`.

    Args:
        transforms (list): A list of image transformations.
        dataset_root (str, optional): The ADK20K dataset directory. Default: None.
        mode (str, optional): A subset of the entire dataset. It should be one of ('train', 'val'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 2

    def __init__(self, transforms, dataset_root=None, mode='train', edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.ignore_index = 255
        self.mode = mode
        self.num_classes = self.NUM_CLASSES
        self.input_width = 224
        self.input_height = 224
        self.std = [0.23, 0.23, 0.23]
        self.mean = [0.485, 0.458, 0.408]

        if mode == 'train':
            path = os.path.join(dataset_root, 'supervisely_face_train_easy.txt')
        else:
            path = os.path.join(dataset_root, 'supervisely_face_test_easy.txt')
        with open(path, 'r') as f:
            files = f.readlines()
        files = ["/".join(file.split('/')[1:]) for file in files]
        img_files = [os.path.join(dataset_root, file).strip() for file in files]
        label_files = [
            os.path.join(dataset_root, file.replace('/img/', '/ann/')).strip()
            for file in files
        ]

        self.file_list = [[
            img_path, label_path
        ] for img_path, label_path in zip(img_files, label_files)]

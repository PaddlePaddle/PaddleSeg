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

from paddleseg3d.cvlibs import manager
from paddleseg3d.transforms import Compose
from paddleseg3d.datasets import MedicalDataset

URL = ' '  # todo: add coronavirus url


@manager.DATASETS.add_component
class LungCoronavirus(MedicalDataset):
    """
    The Lung cornavirus dataset is ...(todo: add link and description)

    Args:
        dataset_root (str): The dataset directory. Default: None
        result_root(str): The directory to save the result file. Default: None
        transforms (list): Transforms for image.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val'). Default: 'train'.

        Examples:

            transforms=[]
            dataset_root = "data/lung_coronavirus/lung_coronavirus_phase0/"
            dataset = LungCoronavirus(dataset_root=dataset_root, transforms=[], num_classes=3, mode="train")

            for data in dataset:
                img, label = data
                print(img.shape, label.shape) # (1, 128, 128, 128) (128, 128, 128)
                print(np.unique(label))

    """

    def __init__(self,
                 dataset_root=None,
                 result_dir=None,
                 transforms=None,
                 num_classes=None,
                 mode='train',
                 ignore_index=255):
        super(LungCoronavirus, self).__init__(
            dataset_root,
            result_dir,
            transforms,
            num_classes,
            mode,
            ignore_index,
            data_URL=URL)
        self.num_classes = num_classes

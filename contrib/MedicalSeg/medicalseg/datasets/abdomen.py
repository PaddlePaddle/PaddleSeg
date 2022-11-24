# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np

from medicalseg.datasets import MedicalDataset
from medicalseg.cvlibs import manager

from medicalseg.utils import loss_computation


@manager.DATASETS.add_component
class Abdomen(MedicalDataset):
    def __init__(self,
                 dataset_root,
                 result_dir,
                 transforms,
                 num_classes,
                 mode,
                 dataset_json_path=""):
        super(Abdomen, self).__init__(
            dataset_root,
            result_dir,
            transforms,
            num_classes,
            mode,
            dataset_json_path=dataset_json_path,
            repeat_times=1)

    def __getitem__(self, idx):

        image_path, label_path = self.file_list[idx]

        image = np.load(image_path)
        label = np.load(label_path)
        if self.mode == "train":
            image = image[np.newaxis, :, :]
            label = label[np.newaxis, :, :]
        else:
            images = image[:, np.newaxis, :, :]
            labels = label[:, np.newaxis, :, :]

        if self.transforms:
            if self.mode == "train":
                image, label = self.transforms(im=image, label=label)
            else:
                image_list = []
                label_list = []
                for i in range(images.shape[0]):
                    image = images[i]
                    label = labels[i]
                    image, label = self.transforms(im=image, label=label)
                    image_list.append(image)
                    label_list.append(label[np.newaxis, :, :, :])
                image = np.concatenate(image_list)
                label = np.concatenate(label_list)

        idx = image_path.split('/')[-1].split('_')[0]
        return image.astype('float32'), label.astype('int64'), idx

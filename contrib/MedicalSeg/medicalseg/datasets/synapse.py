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
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))

import numpy as np

from medicalseg.datasets import MedicalDataset
from medicalseg.cvlibs import manager
from medicalseg.transforms import Compose


@manager.DATASETS.add_component
class Synapse(MedicalDataset):
    def __init__(self,
                 dataset_root,
                 mode,
                 num_classes,
                 result_dir,
                 transforms=None):
        if isinstance(transforms, list):
            transforms = Compose(transforms)
        self.transforms = transforms
        self.mode = mode
        self.sample_list = open(
            os.path.join(dataset_root, self.mode + '.txt')).readlines()
        self.dataset_root = dataset_root
        self.num_classes = num_classes
        self.result_dir = result_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        sample = self.sample_list[idx].strip('\n')
        image_path, label_path = sample.split(' ')

        image = np.load(os.path.join(self.dataset_root, image_path))
        label = np.load(os.path.join(self.dataset_root, label_path))
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
                pass

        return image.astype('float32'), label.astype('int64'), self.sample_list[
            idx].strip('\n').split(" ")[0].split('/')[-1].split('_')[0]

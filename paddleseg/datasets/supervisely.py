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
import copy

import cv2
import numpy as np

from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
from paddleseg.datasets import Dataset
import paddleseg.transforms.functional as F


@manager.DATASETS.add_component
class SUPERVISELY(Dataset):
    """
    Supervise.ly dataset `https://supervise.ly/`.

    Args:
        transforms (list): A list of image transformations.
        dataset_root (str, optional): The ADK20K dataset directory. Default: None.
        mode (str, optional): A subset of the entire dataset. It should be one of ('train', 'val'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 2

    def __init__(self,
                 transforms,
                 transforms2,
                 dataset_root=None,
                 mode='train',
                 edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        if transforms2 is not None:
            self.transforms2 = Compose(transforms2)
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

    def __getitem__(self, item):
        image_path, label_path = self.file_list[item]
        im = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)
        label[label > 1] = 0

        if self.mode == "val":
            im, label = self.transforms(im=im, label=label)
            im = np.float32(im[::-1, :, :])
            im_aug = copy.deepcopy(im)
        else:
            im, label = self.transforms(im=im, label=label)
            # add augmentation
            std = np.array(self.std)[:, np.newaxis, np.newaxis]
            mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
            im_aug = im * std + mean
            im_aug *= 255.0
            im_aug = np.transpose(im_aug, [1, 2, 0]).astype('uint8')
            im_aug = im_aug[:, :, ::-1]

            im_aug, _ = self.transforms2(im_aug)
            im_aug = np.float32(im_aug[::-1, :, :])
            im = np.float32(im[::-1, :, :])

        label = cv2.resize(
            np.uint8(label), (self.input_width, self.input_height),
            interpolation=cv2.INTER_NEAREST)

        # add mask blur
        label = np.uint8(cv2.blur(label, (5, 5)))
        label[label >= 0.5] = 1
        label[label < 0.5] = 0

        # edge = show_edge(label)
        edge_mask = F.mask_to_binary_edge(
            label, radius=4, num_classes=self.num_classes)
        edge_mask = np.transpose(edge_mask, [1, 2, 0]).squeeze(axis=-1)
        cv2.imwrite("edge.png", edge_mask * 255)
        im = np.concatenate([im_aug, im])
        if self.mode == "train":
            return im, label, edge_mask
        else:
            return im, label

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


@manager.DATASETS.add_component
class Synapse(paddle.io.Dataset):
    """
        Synapse dataset `https://www.synapse.org/#!Synapse:syn3193805/wiki/217789`.
        The folder structure is as follow:

            Training
            |
            |--img
            |  |--img0001.nii.gz
            |  |--img0002.nii.gz
            |  |--img0003.nii.gz
            |..............................
            |--label
            |  |--label0001.nii.gz
            |  |--label0002.nii.gz
            |  |--label0003.nii.gz
        Args:
            dataset_root (str): The dataset directory. Default: None
            result_root(str): The directory to save the result file. Default: None
            transforms (list): Transforms for image.
            num_classes(int): The number of classes of the dataset.
            mode (str, optional): Which part of dataset to use. It is one of ('train', 'val'). Default: 'train'.
            dataset_json_path (str, optional): Currently, this argument is not used.

            Examples:

                transforms=[]
                dataset_root = "SynaDataset/preprocessed"
                dataset = SynaDataset(dataset_root=dataset_root, transforms=[], num_classes=9",
                 mode="train")

                for data in dataset:
                    img, label = data
                    print(img.shape, label.shape) # (1, 1 , 1, 224, 224) (1,9,1,224,224)
                    print(np.unique(label))

        """

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

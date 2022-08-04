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
class ACDCDataset(paddle.io.Dataset):
    """
    The MRISpineSeg dataset is come from the MRI Spine Seg competition

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
                 anno_path=None,
                 epoch_batches=1000,
                 mode='train',
                 dataset_json_path=""):
        super(ACDCDataset, self).__init__()
        self.dataset_dir = dataset_root if dataset_root is not None else ''
        self.anno_path = anno_path
        self.transforms = Compose(transforms)
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.epoch_batches=epoch_batches
        self.dataset_json_path = dataset_json_path
        with open(os.path.join(self.dataset_dir,self.anno_path), 'r') as f:
            for line in f:
                items = line.strip().split()
                image_path = os.path.join(self.dataset_dir, items[0])
                grt_path = os.path.join(self.dataset_dir, items[1])
                self.file_list.append([image_path, grt_path])

    def __getitem__(self, idx):
        # print("indx:{}".format(idx))
        if self.mode=="train":
            idx=idx%len(self.file_list)
        # print("indx:{}".format(idx))
        image_path, label_path = self.file_list[idx]
        im, label = self.transforms(im=image_path, label=label_path)

        return im.astype("float32"), label, self.file_list[idx][0]  # npy file name

    def __len__(self):
        if self.mode=="train":
            return self.epoch_batches
        return len(self.file_list)


if __name__ == "__main__":
    dataset = ACDCDataset(
        dataset_dir=r"F:\task01",
        transforms=[],
        num_classes=4,
        anno_path="val_fold0.txt",
        mode='train')
    for item in dataset:
        # print(item)
        img, label,filename = item
        print("img.shape:{}".format(img.shape))
        print("label.shape:{}".format(label.shape))

        if np.any(np.isnan(img)):
            print(img.dtype, label.dtype)  # (1, 128, 128, 12) float32, int64

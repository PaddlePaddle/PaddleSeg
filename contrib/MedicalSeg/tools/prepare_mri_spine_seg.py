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
"""
The file structure is as following:
MRSpineSeg
|--MRI_train.zip
|--MRI_spine_seg_raw
│   └── MRI_train
│       └── train
│           ├── Mask
│           └── MR
├── MRI_spine_seg_phase0
│   ├── images
│   ├── labels
│   │   ├── Case129.npy
│   │   ├── ...
│   ├── train_list.txt
│   └── val_list.txt
└── MRI_train.zip

support:
1. download and uncompress the file.
2. save the normalized data as the above format.
3. split the training data and save the split result in train_list.txt and val_list.txt (we use all the data for training, since this is trainsplit)

"""
import os
import sys
import time
import zipfile
import functools
import numpy as np

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from prepare import Prep
from preprocess_utils import resample, Normalize, label_remap

urls = {
    "MRI_train.zip":
    "https://bj.bcebos.com/v1/ai-studio-online/4e1d24412c8b40b082ed871775ea3e090ce49a83e38b4dbd89cc44b586790108?responseContentDisposition=attachment%3B%20filename%3Dtrain.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2021-04-15T02%3A23%3A20Z%2F-1%2F%2F999e2a80240d9b03ce71b09418b3f2cb1a252fd9cbdff8fd889f7ab21fe91853",
}


class Prep_lung_coronavirus(Prep):
    def __init__(self):
        self.dataset_root = "data/MRSpineSeg"
        self.phase_path = os.path.join(self.dataset_root,
                                       "MRI_spine_seg_phase0_class3_big_12/")
        super().__init__(
            phase_path=self.phase_path, dataset_root=self.dataset_root)

        self.raw_data_path = os.path.join(self.dataset_root,
                                          "MRI_spine_seg_raw/")
        self.image_dir = os.path.join(self.raw_data_path, "MRI_train/train/MR")
        self.label_dir = os.path.join(self.raw_data_path,
                                      "MRI_train/train/Mask")
        self.urls = urls

    def convert_path(self):
        """convert nii.gz file to numpy array in the right directory"""

        print(
            "Start convert images to numpy array using {}, please wait patiently"
            .format(self.gpu_tag))
        time1 = time.time()
        self.load_save(
            self.image_dir,
            save_path=self.image_path,
            preprocess=[
                functools.partial(Normalize, min_val=0, max_val=2650),
                functools.partial(
                    resample, new_shape=[512, 512, 12],
                    order=1)  # original shape is (1008, 1008, 12)
            ],
            valid_suffix=("nii.gz"),
            filter_key=None)

        self.load_save(
            self.label_dir,
            self.label_path,
            preprocess=[
                functools.partial(resample, new_shape=[512, 512, 12], order=0)
            ],
            valid_suffix=("nii.gz"),
            filter_key=None,
            tag="label")

        print("The preprocess time on {} is {}".format(self.gpu_tag,
                                                       time.time() - time1))

    def generate_txt(self, train_split=1.0):
        """generate the train_list.txt and val_list.txt"""

        txtname = [
            os.path.join(self.phase_path, 'train_list.txt'),
            os.path.join(self.phase_path, 'val_list.txt')
        ]

        image_files = os.listdir(self.image_path)
        label_files = [
            name.replace("Case", "mask_case") for name in image_files
        ]

        self.split_files_txt(
            txtname[0], image_files, label_files, train_split=train_split)
        self.split_files_txt(
            txtname[1], image_files, label_files, train_split=train_split)


if __name__ == "__main__":
    prep = Prep_lung_coronavirus()
    prep.uncompress_file(num_zipfiles=1)
    prep.convert_path()
    prep.generate_txt()

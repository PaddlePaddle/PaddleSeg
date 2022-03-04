#  HU_min -1250，HU_max 250
# resample size 128, 128, 128，整CT直接resize，order=1
# LUNA的label mapping就是上午给你的那个{1:0, 4:3, 5:4, 6:5, 7:1, 8:2, 512:0, 516:0, 517:0, 518:0, 519:0, 520:0}
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
luna16_lobe51
|--luna16_lobe51_raw
│   ├── images
│   │   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd
│   │   ├── ...
│   ├── annotations
│   │   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059_LobeSegmentation.nrrd
│   │   ├── ...
├── luna16_lobe51_phase0
│   ├── images
│   │   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.npy
│   ├── labels
│   │   ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.898642529028521482602829374444_LobeSegmentation.npy
│   │   ├── ...
│   ├── train_list.txt
│   └── val_list.txt

support:
1. download and uncompress the file.
2. save the data as the above format.

"""
import os
import sys
import glob
import time
import random
import zipfile
import functools
import numpy as np
import nibabel as nib

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             ".."))

from prepare import Prep
from preprocess_utils import HUNorm, resample, label_remap

urls = {
    "annotation.zip": "",
    "images.zip": ""
}  # TODO: Add urls and test uncompress file as aforementioned format


class Prep_luna(Prep):

    def __init__(self):
        self.dataset_root = "data/luna16_lobe51"
        self.phase_path = os.path.join(self.dataset_root,
                                       "luna16_lobe51_test/")
        super().__init__(phase_path=self.phase_path,
                         dataset_root=self.dataset_root)

        self.raw_data_path = os.path.join(self.dataset_root,
                                          "luna16_lobe51_raw/")
        self.image_dir = os.path.join(self.raw_data_path, "images")
        self.label_dir = os.path.join(self.raw_data_path, "annotations")
        self.urls = urls

    def convert_path(self):
        """convert nii.gz file to numpy array in the right directory"""

        print(
            "Start convert images to numpy array using {}, please wait patiently"
            .format(self.gpu_tag))
        time1 = time.time()
        self.load_save(self.image_dir,
                       save_path=self.image_path,
                       preprocess=[
                           functools.partial(HUNorm, HU_min=-1250, HU_max=250),
                           functools.partial(resample,
                                             new_shape=[128, 128, 128],
                                             order=1)
                       ],
                       valid_suffix=('mhd'),
                       filter_key=None)

        self.load_save(self.label_dir,
                       self.label_path,
                       preprocess=[
                           functools.partial(resample,
                                             new_shape=[128, 128, 128],
                                             order=0),
                           functools.partial(label_remap,
                                             map_dict={
                                                 1: 0,
                                                 4: 2,
                                                 5: 2,
                                                 6: 2,
                                                 7: 1,
                                                 8: 1,
                                                 512: 0,
                                                 516: 0,
                                                 517: 0,
                                                 518: 0,
                                                 519: 0,
                                                 520: 0
                                             })
                       ],
                       valid_suffix=('nrrd'),
                       filter_key=None,
                       tag="label")

        print("The preprocess time on {} is {}".format(
            "GPU" if self.gpu_tag else "CPU",
            time.time() - time1))

    def generate_txt(self):
        """generate the train_list.txt and val_list.txt"""

        txtname = [
            os.path.join(self.phase_path, 'train_list.txt'),
            os.path.join(self.phase_path, 'val_list.txt')
        ]

        label_files = os.listdir(self.label_path)
        image_files = [
            name.replace("_LobeSegmentation", "") for name in label_files
        ]

        self.split_files_txt(txtname[0],
                             image_files,
                             label_files,
                             train_split=45)
        self.split_files_txt(txtname[1],
                             image_files,
                             label_files,
                             train_split=45)


if __name__ == "__main__":
    prep = Prep_luna()
    # prep.uncompress_file(num_zipfiles=4)
    prep.convert_path()
    prep.generate_txt()

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
lung_coronavirus
|--20_ncov_scan.zip
|--infection.zip
|--lung_infection.zip
|--lung_mask.zip
|--lung_coronavirus_raw
│   ├── 20_ncov_scan
│   │   ├── coronacases_org_001.nii.gz
│   │   ├── ...
│   ├── infection_mask
│   ├── lung_infection
│   ├── lung_mask
├── lung_coronavirus_phase0
│   ├── images
│   ├── labels
│   │   ├── coronacases_001.npy
│   │   ├── ...
│   │   └── radiopaedia_7_85703_0.npy
│   ├── train_list.txt
│   └── val_list.txt
support:
1. download and uncompress the file.
2. save the data as the above format.
3. split the training data and save the split result in train_list.txt and val_list.txt

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
from preprocess_utils import HUNorm, resample

urls = {
    "lung_infection.zip":
    "https://bj.bcebos.com/v1/ai-studio-online/432237969243497caa4d389c33797ddb2a9fa877f3104e4a9a63bd31a79e4fb8?responseContentDisposition=attachment%3B%20filename%3DLung_Infection.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2020-05-10T03%3A42%3A16Z%2F-1%2F%2Faccd5511d56d7119555f0e345849cca81459d3783c547eaa59eb715df37f5d25",
    "lung_mask.zip":
    "https://bj.bcebos.com/v1/ai-studio-online/96f299c5beb046b4a973fafb3c39048be8d5f860bd0d47659b92116a3cd8a9bf?responseContentDisposition=attachment%3B%20filename%3DLung_Mask.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2020-05-10T03%3A41%3A14Z%2F-1%2F%2Fb8e23810db1081fc287a1cae377c63cc79bac72ab0fb835d48a46b3a62b90f66",
    "infection_mask.zip":
    "https://bj.bcebos.com/v1/ai-studio-online/2b867932e42f4977b46bfbad4fba93aa158f16c79910400b975305c0bd50b638?responseContentDisposition=attachment%3B%20filename%3DInfection_Mask.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2020-05-10T03%3A42%3A37Z%2F-1%2F%2Fabd47aa33ddb2d4a65555795adef14826aa68b20c3ee742dff2af010ae164252",
    "20_ncov_scan.zip":
    "https://bj.bcebos.com/v1/ai-studio-online/12b02c4d5f9d44c5af53d17bbd4f100888b5be1dbc3d40d6b444f383540bd36c?responseContentDisposition=attachment%3B%20filename%3D20_ncov_scan.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2020-05-10T14%3A54%3A21Z%2F-1%2F%2F1d812ca210f849732feadff9910acc9dcf98ae296988546115fa7b987d856b85"
}


class Prep_lung_coronavirus(Prep):
    def __init__(self):
        self.dataset_root = "data/lung_coronavirus_test"
        self.phase_path = os.path.join(self.dataset_root,
                                       "lung_coronavirus_phase0/")
        super().__init__(
            phase_path=self.phase_path, dataset_root=self.dataset_root)

        self.raw_data_path = os.path.join(self.dataset_root,
                                          "lung_coronavirus_raw/")
        self.image_dir = os.path.join(self.raw_data_path, "20_ncov_scan")
        self.label_dir = os.path.join(self.raw_data_path, "lung_mask")
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
                HUNorm,
                functools.partial(
                    resample, new_shape=[128, 128, 128], order=1)
            ],
            valid_suffix=("nii.gz"),
            filter_key=None)

        self.load_save(
            self.label_dir,
            self.label_path,
            preprocess=[
                functools.partial(
                    resample, new_shape=[128, 128, 128], order=0),
            ],
            valid_suffix=("nii.gz"),
            filter_key=None,
            tag="label")

        print("The preprocess time on {} is {}".format(self.gpu_tag,
                                                       time.time() - time1))

    def generate_txt(self, train_split=0.75):
        """generate the train_list.txt and val_list.txt"""

        txtname = [
            os.path.join(self.phase_path, 'train_list.txt'),
            os.path.join(self.phase_path, 'val_list.txt')
        ]

        image_files = os.listdir(self.image_path)
        label_files = [
            name.replace("_org_covid-19-pneumonia-",
                         "_").replace("-dcm", "").replace("_org_", "_")
            for name in image_files
        ]

        self.split_files_txt(
            txtname[0], image_files, label_files, train_split=train_split)
        self.split_files_txt(
            txtname[1], image_files, label_files, train_split=train_split)


if __name__ == "__main__":

    prep = Prep_lung_coronavirus()
    # prep.uncompress_file(num_zipfiles=4)
    prep.convert_path()
    prep.generate_txt()

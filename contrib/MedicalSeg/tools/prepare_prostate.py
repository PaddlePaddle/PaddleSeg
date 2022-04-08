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
import zipfile
import functools
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from prepare import Prep
from preprocess_utils import resample, normalize, label_remap
from medicalseg.utils import wrapped_partial

urls = {
    "Promise12": {
        "Promise12": ""
    },
    "Prostate_mri": {
        "Prostate_mri": ""
    },  # https://drive.google.com/file/d/1TtrjnlnJ1yqr5m4LUGMelKTQXtvZaru-/view?usp=sharing
}

dataset_addr = {
    "Promise12": {
        "dataset_root": "data/Promise12",
        "raw_dataset_dir": "Promise12_raw",
        "images_dir":
        ("prostate/TrainingData_Part1", "prostate/TrainingData_Part2",
         "prostate/TrainingData_Part3"),
        "labels_dir": ("prostate/TrainingData_Part1",
                       "prostate/TrainingData_Part2",
                       "prostate/TrainingData_Part3"),
        "images_dir_test": "prostate/TestData",
        "phase_dir": "Promise12_phase0/",
        "urls": urls["Promise12"],
        "valid_suffix": ("mhd", "mhd"),
        "filter_key": ({
            "segmentation": False
        }, {
            "segmentation": True
        }),
        "uncompress_params": {
            "format": "zip",
            "num_files": 1
        }
    },
    "Prostate_mri": {
        "dataset_root": "data/Prostate_mri",
        "raw_dataset_dir": "Prostate_mri_raw",
        "images_dir": ("Processed_data_nii/BIDMC", "Processed_data_nii/BMC",
                       "Processed_data_nii/HK", "Processed_data_nii/I2CVB",
                       "Processed_data_nii/RUNMC", "Processed_data_nii/UCL"),
        "labels_dir": ("Processed_data_nii/BIDMC", "Processed_data_nii/BMC",
                       "Processed_data_nii/HK", "Processed_data_nii/I2CVB",
                       "Processed_data_nii/RUNMC", "Processed_data_nii/UCL"),
        "phase_dir": "Prostate_mri_phase0/",
        "urls": urls["Prostate_mri"],
        "valid_suffix": ("nii.gz", "nii.gz"),
        "filter_key": ({
            "segmentation": False
        }, {
            "segmentation": True
        }),
        "uncompress_params": {
            "format": "zip",
            "num_files": 1
        }
    }
}

dataset_profile = {
    "Promise12": {
        "modalities": ('MRI-T2', ),
        "labels": {
            0: "Background",
            1: "prostate"
        },
        "dataset_name": "Promise12",
        "dataset_description":
        "These cases include a transversal T2-weighted MR image of the prostate. The training set is a representative set of the types of MR images acquired in a clinical setting. The data is multi-center and multi-vendor and has different acquistion protocols (e.g. differences in slice thickness, with/without endorectal coil). The set is selected such that there is a spread in prostate sizes and appearance. For each of the cases in the training set, a reference segmentation is also included.",
        "license_desc": "",
        "dataset_reference": "https://promise12.grand-challenge.org/Details/"
    },
    "Prostate_mri": {
        "modalities": ('MRI-T2', ),
        "labels": {
            0: "Background",
            1: "prostate"
        },
        "dataset_name": "Prostate_mri",
        "dataset_description":
        "This is a well-organized multi-site dataset for prostate MRI segmentation, which contains prostate T2-weighted MRI data (with segmentation mask) collected from six different data sources out of three public datasets. ",
        "license_desc": "",
        "dataset_reference": "https://liuquande.github.io/SAML/"
    }
}


class Prep_prostate(Prep):
    def __init__(self,
                 dataset_root="data/TemDataSet",
                 raw_dataset_dir="TemDataSet_seg_raw/",
                 images_dir="train_imgs",
                 labels_dir="train_labels",
                 phase_dir="phase0",
                 urls=None,
                 valid_suffix=("nii.gz", "nii.gz"),
                 filter_key=(None, None),
                 uncompress_params={"format": "zip",
                                    "num_files": 1},
                 images_dir_test=""):

        super().__init__(dataset_root, raw_dataset_dir, images_dir, labels_dir,
                         phase_dir, urls, valid_suffix, filter_key,
                         uncompress_params, images_dir_test)

        self.preprocess={"images":[           # todo: make params set automatically
                        normalize,
                        wrapped_partial(
                            resample, new_shape=[512, 512, 24],
                            order=1)],
                        "labels":[
                        wrapped_partial(
                            resample, new_shape=[512, 512, 24], order=0)],
                        "images_test":[normalize,]}

    def generate_txt(self, split=1.0):
        """generate the train_list.txt and val_list.txt"""

        txtname = [
            os.path.join(self.phase_path, 'train_list.txt'),
            os.path.join(self.phase_path, 'val_list.txt')
        ]

        if self.image_files_test:
            txtname.append(os.path.join(self.phase_path, 'test_list.txt'))
            test_file_npy = os.listdir(self.image_path_test)

        image_files_npy = os.listdir(self.image_path)
        label_files_npy = [
            name.replace(".npy", "_segmentation.npy")
            for name in image_files_npy  # to have the save order
        ]

        self.split_files_txt(
            txtname[0], image_files_npy, label_files_npy, split=split)
        self.split_files_txt(
            txtname[1], image_files_npy, label_files_npy, split=split)

        self.split_files_txt(txtname[2], test_file_npy)


if __name__ == "__main__":
    # Todo: Prostate_mri have files with same name in different dir, which caused file overlap problem.
    # Todo: MSD_prostate is not supported yet, because it has four channel and resample will have a bug.
    prep = Prep_prostate(**dataset_addr["Promise12"])
    prep.generate_dataset_json(**dataset_profile["Promise12"])
    prep.load_save()
    prep.generate_txt()

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
    "MRI_train.zip":
    "https://bj.bcebos.com/v1/ai-studio-online/4e1d24412c8b40b082ed871775ea3e090ce49a83e38b4dbd89cc44b586790108?responseContentDisposition=attachment%3B%20filename%3Dtrain.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2021-04-15T02%3A23%3A20Z%2F-1%2F%2F999e2a80240d9b03ce71b09418b3f2cb1a252fd9cbdff8fd889f7ab21fe91853",
}


class Prep_mri_spine(Prep):
    def __init__(self):
        super().__init__(
            dataset_root="data/MRSpineSeg",
            raw_dataset_dir="MRI_spine_seg_raw/",
            images_dir="MRI_train/train/MR",
            labels_dir="MRI_train/train/Mask",
            phase_dir="MRI_spine_seg_phase0_class20_big_12/",
            urls=urls,
            valid_suffix=("nii.gz", "nii.gz"),
            filter_key=(None, None),
            uncompress_params={"format": "zip",
                               "num_files": 1})

        self.preprocess = {
            "images": [
                wrapped_partial(
                    normalize, min_val=0, max_val=2650), wrapped_partial(
                        resample, new_shape=[512, 512, 12], order=1)
            ],  # original shape is (1008, 1008, 12)
            "labels":
            [wrapped_partial(
                resample, new_shape=[512, 512, 12], order=0)]
        }

    def generate_txt(self, train_split=1.0):
        """generate the train_list.txt and val_list.txt"""

        txtname = [
            os.path.join(self.phase_path, 'train_list.txt'),
            os.path.join(self.phase_path, 'val_list.txt')
        ]

        image_files_npy = os.listdir(self.image_path)
        label_files_npy = [
            name.replace("Case", "mask_case") for name in image_files_npy
        ]

        self.split_files_txt(txtname[0], image_files_npy, label_files_npy,
                             train_split)
        self.split_files_txt(txtname[1], image_files_npy, label_files_npy,
                             train_split)


if __name__ == "__main__":
    prep = Prep_mri_spine()
    prep.generate_dataset_json(
            modalities=('MRI-T2', ),
            labels={
                0: "Background",
                1: "S",
                2: "L5",
                3: "L4",
                4: "L3",
                5: "L2",
                6: "L1",
                7: "T12",
                8: "T11",
                9: "T10",
                10: "T9",
                11: "L5/S",
                12: "L4/L5",
                13: "L3/L4",
                14: "L2/L3",
                15: "L1/L2",
                16: "T12/L1",
                17: "T11/T12",
                18: "T10/T11",
                19: "T9/T10"
            },
            dataset_name="MRISpine Seg",
            dataset_description="There are 172 training data in the preliminary competition, including MR images and mask labels, 20 test data in the preliminary competition and 23 test data in the  second round competition. The labels of the preliminary competition testset and the second round competition testset are not published, and the results can be evaluated online on this website.",
            license_desc="https://www.spinesegmentation-challenge.com/wp-content/uploads/2021/12/Term-of-use.pdf",
            dataset_reference="https://www.spinesegmentation-challenge.com/", )
    prep.load_save()
    prep.generate_txt()

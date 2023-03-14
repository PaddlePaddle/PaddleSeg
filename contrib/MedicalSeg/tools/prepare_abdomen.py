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
"""
The file structure is as following:
abdomen
|--RawData.zip
|--abdomen_raw
│   ├── RawData
│   │   ├──RawData
│   │   │   ├── Training
│   │   │   │   ├── img
│   │   │   │   │   ├── img0001.nii.gz
│   │   │   │   │   └── ...
│   │   │   │   └── ...
│   │   │   │   ├── label
│   │   │   │   │   ├── img0001.nii.gz
│   │   │   │   │   └── ...
│   │   │   │   └── ...
├── abdomen_phase0
│   ├── images
│   │   ├── img0001-0001.npy
│   │   └── ...
│   ├── labels
│   │   ├── label0001-0001.npy
│   │   └── ...
│   ├── train_list.txt
│   └── val_list.txt
support:
1. download and uncompress the file.
2. save the data as the above format.
3. split the training data and save the split result in train_list.txt and val_list.txt

"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import os.path as osp
import time
import json

import numpy as np
from tqdm import tqdm

from prepare import Prep
from preprocess_utils import HUnorm, ignore_label, join_paths
from medicalseg.utils import wrapped_partial

urls = {"Reg-Training-Training.zip": ""}

label_map = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 0,
    "6": 5,
    "7": 6,
    "8": 7,
    "9": 0,
    "10": 0,
    "11": 8,
    "12": 0,
    "13": 0,
}


class Prep_abdomen(Prep):
    def __init__(self):
        super().__init__(
            dataset_root="data/abdomen",
            raw_dataset_dir="abdomen_raw/",
            images_dir="RawData/RawData/Training/img",
            labels_dir="RawData/RawData/Training/label",
            phase_dir="abdomen_phase0/",
            urls=urls,
            valid_suffix=("nii.gz", "nii.gz"),
            filter_key=(None, None),
            uncompress_params={"format": "zip",
                               "num_files": 1})

        self.preprocess = {
            "images": [
                wrapped_partial(
                    np.clip, a_min=-125, a_max=275), wrapped_partial(
                        HUnorm, HU_min=-125, HU_max=275, multiply_255=False)
            ],
            "labels": [wrapped_partial(
                ignore_label, label_map=label_map)]
        }
        self.train_image_files = []
        self.val_image_files = []
        self.train_label_files = []
        self.val_label_files = []

        self.train_image_files_npy = []
        self.val_image_files_npy = []
        self.train_label_files_npy = []
        self.val_label_files_npy = []
        self.train_val_split()

    def load_save(self, mode='train'):
        """
        preprocess files, transfer to the correct type, and save it to the directory.
        """
        print(
            "Start convert {} images to numpy array using {}, please wait patiently"
            .format(mode, self.gpu_tag))

        tic = time.time()
        if mode == 'train':
            process_files = (self.train_image_files, self.train_label_files)
            target_files = (self.train_image_files_npy,
                            self.train_label_files_npy)
        else:
            process_files = (self.val_image_files, self.val_label_files)
            target_files = (self.val_image_files_npy, self.val_label_files_npy)
        process_tuple = ("images", "labels")

        save_tuple = (self.image_path, self.label_path)

        for i, files in enumerate(process_files):
            pre = self.preprocess[process_tuple[i]]
            savepath = save_tuple[i]

            for f in tqdm(
                    files,
                    total=len(files),
                    desc="preprocessing the {}".format(["images", "labels"][
                        i])):

                f_nps = Prep.load_medical_data(f)[0]
                # xyz to zxy
                f_nps = f_nps.transpose(2, 0, 1)
                if mode == 'train':
                    for volume_idx, f_np in enumerate(f_nps):
                        for op in pre:
                            f_np = op(f_np)
                        filename = osp.basename(f).split(".")[
                            0] + f"-{volume_idx:>04d}.npy"
                        f_np_name = join_paths(savepath, filename)
                        np.save(f_np_name, f_np)
                        target_files[i].append(filename)
                else:
                    for op in pre:
                        f_nps = op(f_nps)
                    filename = osp.basename(f).split(".")[0] + ".npy"
                    f_np_name = join_paths(savepath, filename)
                    np.save(f_np_name, f_nps)
                    target_files[i].append(filename)

        print("The preprocess time on {} is {}".format(self.gpu_tag,
                                                       time.time() - tic))

    def generate_txt(self, train_split=0.6):
        """generate the train_list.txt and val_list.txt"""

        txtname = [
            join_paths(self.phase_path, 'train_list.txt'),
            join_paths(self.phase_path, 'val_list.txt')
        ]
        self.write_txt(txtname[0], self.train_image_files_npy,
                       self.train_label_files_npy)
        self.write_txt(txtname[1], self.val_image_files_npy,
                       self.val_label_files_npy)

    def train_val_split(self, train_split=0.6):
        image_files = np.array(self.image_files)
        label_files = np.array(self.label_files)
        np.random.seed(0)
        state = np.random.get_state()
        np.random.shuffle(image_files)
        np.random.set_state(state)
        np.random.shuffle(label_files)
        train_len = round(len(self.image_files) * train_split)
        self.train_image_files = self.image_files[:train_len]
        self.val_image_files = self.image_files[train_len:]
        self.train_label_files = self.label_files[:train_len]
        self.val_label_files = self.label_files[train_len:]


if __name__ == "__main__":
    prep = Prep_abdomen()
    prep.generate_dataset_json(
        modalities=('CT', ),
        labels={
            0: 'background',
            1: 'spleen',
            2: 'right kidney',
            3: 'left kidney',
            4: 'gallbladder',
            5: 'liver',
            6: 'stomach',
            7: 'aorta',
            8: 'pancreas'
        },
        dataset_name="Abdomen CT scans",
        dataset_description="Under Institutional Review Board (IRB) supervision, 50 abdomen CT scans of were randomly selected from a combination of an ongoing colorectal cancer chemotherapy trial, and a retrospective ventral hernia study.",
        license_desc="https://creativecommons.org/licenses/by/4.0/legalcode",
        dataset_reference="https://www.synapse.org/#!Synapse:syn3193805/wiki/89480",
    )
    prep.load_save(mode='train')
    prep.load_save(mode='val')
    prep.generate_txt()

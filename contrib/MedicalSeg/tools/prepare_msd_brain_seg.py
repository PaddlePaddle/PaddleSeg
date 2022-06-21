# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os.path as osp
import sys
import time

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm

sys.path.append(osp.join(osp.dirname(osp.realpath(__file__)), ""))

from prepare import Prep

tasks = {
    1: {
        "Task01_BrainTumour.tar":
        "https://bj.bcebos.com/v1/ai-studio-online/netdisk/975fea1d4c8549b883b2b4bb7e6a82de84392a6edd054948b46ced0f117fd701?responseContentDisposition=attachment%3B%20filename%3DTask01_BrainTumour.tar&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-01-21T18%3A50%3A30Z%2F-1%2F%2F283ea6f8700c129903e3278ea38a54eac2cf087e7f65197268739371898aa1b3"
    }
}


class PrepMSDBrain(Prep):
    def __init__(self, task_id):
        task_name = list(tasks[task_id].keys())[0].split('.')[0]
        print(f"Preparing task {task_id} {task_name}")
        super().__init__(
            dataset_root=f"data/{task_name}",
            raw_dataset_dir=f"{task_name}_raw/",
            images_dir=f"{task_name}/{task_name}/imagesTr",
            labels_dir=f"{task_name}/{task_name}/labelsTr",
            phase_dir=f"{task_name}_phase0/",
            urls=tasks[task_id],
            valid_suffix=("nii.gz", "nii.gz"),
            filter_key=(None, None),
            uncompress_params={"format": "tar",
                               "num_files": 1})
        self.preprocess = {"images": [], "labels": []}

    def generate_txt(self, train_split=0.8, test_split=0.95):
        """generate the train_list.txt and val_list.txt"""
        txtname = [
            osp.join(self.phase_path, 'train_list.txt'),
            osp.join(self.phase_path, 'val_list.txt'),
            osp.join(self.phase_path, 'test_list.txt')
        ]
        image_files_npy = os.listdir(self.image_path)
        label_files_npy = os.listdir(self.label_path)
        self.split_files_txt(txtname[0], image_files_npy, label_files_npy,
                             train_split, test_split)
        self.split_files_txt(txtname[1], image_files_npy, label_files_npy,
                             train_split, test_split)
        self.split_files_txt(txtname[2], image_files_npy, label_files_npy,
                             train_split, test_split)

    def split_files_txt(self,
                        txt,
                        image_files,
                        label_files=None,
                        split=None,
                        testsplit=None):
        split = int(split * len(image_files))
        testsplit = int(testsplit * len(image_files))
        if "train" in txt:
            image_names = image_files[:split]
            label_names = label_files[:split]
        elif "val" in txt:
            # set the valset to 20% of images if all files need to be used in training
            image_names = image_files[split:testsplit]
            label_names = label_files[split:testsplit]
        elif "test" in txt:
            image_names = image_files[testsplit:]
            label_names = label_files[testsplit:]
        else:
            raise NotImplementedError(
                "The txt split except for train.txt, val.txt and test.txt is not implemented yet."
            )
        self.write_txt(txt, image_names, label_names)

    @staticmethod
    def load_medical_data(f):
        """
        load data of different format into numpy array, return data is in xyz
        f: the complete path to the file that you want to load
        """
        filename = osp.basename(f).lower()
        images = []
        # validate nii.gz on lung and mri with correct spacing_resample
        if filename.endswith((".nii", ".nii.gz", ".dcm")):
            if "radiopaedia" in filename or "corona" in filename:
                f_nps = [nib.load(f).get_fdata(dtype=np.float32)]
            else:
                itkimage = sitk.ReadImage(f)
                if itkimage.GetDimension() == 4:
                    images = [itkimage]
                else:
                    images = [itkimage]
                f_nps = [sitk.GetArrayFromImage(img) for img in images]
        return f_nps

    def load_save(self):
        """
        preprocess files, transfer to the correct type, and save it to the directory.
        """
        print(
            "Start convert images to numpy array using {}, please wait patiently"
            .format(self.gpu_tag))
        tic = time.time()
        process_files = (self.image_files, self.label_files)
        process_tuple = ("images", "labels")
        save_tuple = (self.image_path, self.label_path)
        for i, files in enumerate(process_files):
            pre = self.preprocess[process_tuple[i]]
            savepath = save_tuple[i]
            for f in tqdm(
                    files,
                    total=len(files),
                    desc="preprocessing the {}".format(
                        ["images", "labels", "images_test"][i])):
                # load data will transpose the image from "zyx" to "xyz"
                spacing = (1, 1, 1)
                f_nps = self.load_medical_data(f)
                for volume_idx, f_np in enumerate(f_nps):
                    for op in pre:
                        if op.__name__ == "resample":
                            f_np, new_spacing = op(
                                f_np,
                                spacing=spacing)  # (960, 15, 960) if transpose
                        else:
                            f_np = op(f_np)
                    f_np = f_np.astype("float32") if i == 0 else f_np.astype(
                        "int32")
                    volume_idx = "" if len(f_nps) == 1 else f"-{volume_idx}"
                    np.save(
                        os.path.join(
                            savepath,
                            osp.basename(f).split(".")[0] + volume_idx), f_np)
        print("The preprocess time on {} is {}".format(self.gpu_tag,
                                                       time.time() - tic))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Please provide task id. Example usage: \n\t python tools/prepare_msd.py 1 # for preparing MSD task 1"
        )
    try:
        task_id = int(sys.argv[1])
    except ValueError:
        print(
            f"Expecting number as command line argument, got {sys.argv[1]}.  Example usage: \n\t python tools/prepare_msd.py 1 # for preparing MSD task 1"
        )
    prep = PrepMSDBrain(task_id)
    prep.load_save()
    prep.generate_txt()

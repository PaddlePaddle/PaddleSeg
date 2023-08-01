#    Copyright 2022 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import nibabel as nib
import shutil
import os.path as osp
from preprocess_utils.file_and_folder_operations import *
from preprocess_utils.geometry import *
from preprocess_utils.path_utils import join_paths

from tqdm import tqdm

sys.path.append(osp.join(osp.dirname(osp.realpath(__file__)), ""))


class PrepACDC():
    def __init__(self,
                 dataset_root=f"data/ACDCDataset",
                 raw_dataset_dir=f"training/",
                 clean_dataset_dir=f"clean_data",
                 phase_dir=f"ACDCDataset_phase0"):
        super().__init__()

        self.folder = raw_dataset_dir
        self.clean_folder = join_paths(dataset_root, clean_dataset_dir)
        self.phase_path = join_paths(dataset_root, phase_dir)

    def generate_txt(self, split=0.2):
        """generate the train_list.txt and val_list.txt"""

        txtname = [
            join_paths(self.phase_path, 'train_list.txt'),
            join_paths(self.phase_path, 'val_list.txt')
        ]
        val_len = int(split * len(self.filenames))

        with open(txtname[0], "w") as f:
            for filename in self.filenames[:-val_len]:
                f.write("images/{}.npy labels/{}.npy\n".format(filename,
                                                               filename))
        with open(txtname[1], "w") as f:
            for filename in self.filenames[-val_len:]:

                f.write("images/{}.npy labels/{}.npy\n".format(filename,
                                                               filename))

    def load_save(self, new_spacing):
        self.image_path = join_paths(self.phase_path, "images")
        self.label_path = join_paths(self.phase_path, "labels")
        maybe_mkdir_p(self.image_path)
        maybe_mkdir_p(self.label_path)
        data_lists = os.listdir(join_paths(self.clean_folder, "imagesTr"))
        self.filenames = [filename.split(".")[0] for filename in data_lists]
        for filename in tqdm(data_lists):
            nimg = nib.load(join_paths(self.clean_folder, "imagesTr", filename))
            nlabel = nib.load(
                join_paths(self.clean_folder, "labelsTr", filename))
            data_arrary = nimg.get_data()
            label_array = nlabel.get_data()
            original_spacing = nimg.header["pixdim"][1:4]
            assert data_arrary.shape == label_array.shape
            shape = data_arrary.shape
            new_shape = np.round(((np.array(original_spacing) /
                                   np.array(new_spacing)).astype(float) *
                                  np.array(shape))).astype(int)
            new_data_array = resize_image(data_arrary, new_shape)
            new_label_array = resize_segmentation(label_array, new_shape)
            #将数据从hwd转化为dhw
            new_data_array = np.transpose(new_data_array, [2, 0, 1])
            new_label_array = np.transpose(new_label_array, [2, 0, 1])
            np.save(
                join_paths(self.image_path,
                           filename.replace(r".nii.gz", '.npy')),
                new_data_array)
            np.save(
                join_paths(self.label_path,
                           filename.replace(r".nii.gz", '.npy')),
                new_label_array)

    def clean_raw_data(self):

        maybe_mkdir_p(join(self.clean_folder, "imagesTr"))
        maybe_mkdir_p(join(self.clean_folder, "labelsTr"))

        # train
        all_train_files = []
        patient_dirs_train = subfolders(self.folder, prefix="patient")
        for p in patient_dirs_train:
            current_dir = p
            data_files_train = [
                i for i in subfiles(
                    current_dir, suffix=".nii.gz")
                if i.find("_gt") == -1 and i.find("_4d") == -1
            ]
            corresponding_seg_files = [
                i[:-7] + "_gt.nii.gz" for i in data_files_train
            ]
            for d, s in zip(data_files_train, corresponding_seg_files):
                patient_identifier = os.path.split(d)[1][:-7]
                all_train_files.append(patient_identifier + "_0000.nii.gz")
                shutil.copy(d,
                            join(self.clean_folder, "imagesTr",
                                 patient_identifier + "_0000.nii.gz"))
                shutil.copy(s,
                            join(self.clean_folder, "labelsTr",
                                 patient_identifier + "_0000.nii.gz"))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        prep = PrepACDC(raw_dataset_dir=sys.argv[1])
    else:
        prep = PrepACDC()
    new_spacing = [1.52, 1.52, 6.35]
    prep.clean_raw_data()
    prep.load_save(new_spacing)
    prep.generate_txt()

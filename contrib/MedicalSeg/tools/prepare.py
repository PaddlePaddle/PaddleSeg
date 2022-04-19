# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
File: prepare.py
This is the prepare class for all relavent prepare file

support:
1. download and uncompress the file.
2. save the data as the above format.
3. read the preprocessed data into train.txt and val.txt

"""
import os
import os.path as osp
import sys
import nrrd
import time
import glob
import argparse
import zipfile
import collections
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
import json

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from medicalseg.utils import get_image_list
from tools.preprocess_utils import uncompressor, global_var, add_qform_sform


class Prep:
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
        """
        Create proprosessor for medical dataset.
        Folder structure:
            dataset_root
            ├── raw_dataset_dir
            │   ├── image_dir
            │   ├── labels_dir  
            │   ├── images_dir_test       
            ├── phase_dir
            │   ├── images
            │   ├── labels
            │   ├── train_list.txt
            │   └── val_list.txt
            ├── archive_1.zip
            ├── archive_2.zip
            └── ... archives ...
        Args:
            urls (dict): Urls to download dataset archive. Key will be used as archive name.
            valid_suffix(tuple):  Only files with the assigned suffix will be considered. The first is the suffix for image, and the other is for label.
            filter_key(tuple): Only files containing the filter_key the will be considered.
        """
        # combine all paths
        self.dataset_root = dataset_root
        self.phase_path = os.path.join(self.dataset_root, phase_dir)
        self.raw_data_path = os.path.join(self.dataset_root, raw_dataset_dir)
        self.dataset_json_path = os.path.join(
            self.raw_data_path,
            "dataset.json")  # save the dataset.json to raw path
        self.image_path = os.path.join(self.phase_path, "images")
        self.label_path = os.path.join(self.phase_path, "labels")
        os.makedirs(self.dataset_root, exist_ok=True)
        os.makedirs(self.phase_path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.label_path, exist_ok=True)
        self.gpu_tag = "GPU" if global_var.get_value('USE_GPU') else "CPU"
        self.urls = urls

        if osp.exists(self.raw_data_path):
            print(
                f"raw_dataset_dir {self.raw_data_path} exists, skipping uncompress. To uncompress again, remove this directory"
            )
        else:
            self.uncompress_file(
                num_files=uncompress_params["num_files"],
                form=uncompress_params["format"])

        self.image_files_test = None
        if len(images_dir_test
               ) != 0:  # test image filter is the same as training image
            self.image_files_test = get_image_list(
                os.path.join(self.raw_data_path, images_dir_test),
                valid_suffix[0], filter_key[0])
            self.image_files_test.sort()
            self.image_path_test = os.path.join(self.phase_path, 'images_test')
            os.makedirs(self.image_path_test, exist_ok=True)

        # Load the needed file with filter
        if isinstance(images_dir, tuple):
            self.image_files = []
            self.label_files = []
            for i in range(len(images_dir)):
                self.image_files += get_image_list(
                    os.path.join(self.raw_data_path, images_dir[i]),
                    valid_suffix[0], filter_key[0])
                self.label_files += get_image_list(
                    os.path.join(self.raw_data_path, labels_dir[i]),
                    valid_suffix[1], filter_key[1])
        else:
            self.image_files = get_image_list(
                os.path.join(self.raw_data_path, images_dir), valid_suffix[0],
                filter_key[0])
            self.label_files = get_image_list(
                os.path.join(self.raw_data_path, labels_dir), valid_suffix[1],
                filter_key[1])

        self.image_files.sort()
        self.label_files.sort()

    def uncompress_file(self, num_files, form):
        uncompress_tool = uncompressor(
            download_params=(self.urls, self.dataset_root, True))
        """unzip all the file in the root directory"""
        files = glob.glob(os.path.join(self.dataset_root, "*.{}".format(form)))

        assert len(files) == num_files, "The file directory should include {} compressed files, but there is only {}".format(num_files, len(files))

        for f in files:
            extract_path = os.path.join(self.raw_data_path,
                                        f.split("/")[-1].split('.')[0])
            uncompress_tool._uncompress_file(
                f, extract_path, delete_file=False, print_progress=True)

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
                    slicer = sitk.ExtractImageFilter()
                    s = list(itkimage.GetSize())
                    s[-1] = 0
                    slicer.SetSize(s)
                    for slice_idx in range(itkimage.GetSize()[-1]):
                        slicer.SetIndex([0, 0, 0, slice_idx])
                        sitk_volume = slicer.Execute(itkimage)
                        images.append(sitk_volume)
                else:
                    images = [itkimage]
                images = [sitk.DICOMOrient(img, 'LPS') for img in images]
                f_nps = [sitk.GetArrayFromImage(img) for img in images]

        elif filename.endswith(
            (".mha", ".mhd", "nrrd"
             )):  # validate mhd on lung and mri with correct spacing_resample
            itkimage = sitk.DICOMOrient(sitk.ReadImage(f), 'LPS')
            f_np = sitk.GetArrayFromImage(itkimage)
            if f_np.ndim == 4:
                f_nps = [f_np[:, :, :, idx] for idx in range(f_np.shape[3])]
            else:
                f_nps = [f_np]
            f_nps = [np.transpose(f_np, [1, 2, 0]) for f_np in f_nps]
        elif filename.endswith(".raw"):
            raise RuntimeError(
                f"Received {f}. Please only provide path to .mhd file, not to .raw file"
            )

        else:
            raise NotImplementedError

        return f_nps

    def load_save(self):
        """
        preprocess files, transfer to the correct type, and save it to the directory.
        """
        print(
            "Start convert images to numpy array using {}, please wait patiently"
            .format(self.gpu_tag))

        tic = time.time()
        with open(self.dataset_json_path, 'r', encoding='utf-8') as f:
            dataset_json_dict = json.load(f)

        if self.image_files_test:
            process_files = (self.image_files, self.label_files,
                             self.image_files_test)
            process_tuple = ("images", "labels", "images_test")
            save_tuple = (self.image_path, self.label_path,
                          self.image_path_test)
        else:
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
                spacing = dataset_json_dict["training"][
                            osp.basename(f).split(".")[0]]["spacing"] if i == 0 else None
                f_nps = Prep.load_medical_data(f)

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

                if i == 0:
                    dataset_json_dict["training"][osp.basename(f).split(".")[
                        0]]["spacing_resample"] = new_spacing

        with open(self.dataset_json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_json_dict, f, ensure_ascii=False, indent=4)

        print("The preprocess time on {} is {}".format(self.gpu_tag,
                                                       time.time() - tic))

    def convert_path(self):
        """convert nii.gz file to numpy array in the right directory"""
        raise NotImplementedError

    def generate_txt(self):
        """generate the train_list.txt and val_list.txt"""
        raise NotImplementedError

    # TODO add data visualize method, such that data can be checked every time after preprocess.
    def visualize(self):
        pass
        # imga = Image.fromarray(np.int8(imga))
        # #当要保存的图片为灰度图像时，灰度图像的 numpy 尺度是 [1, h, w]。需要将 [1, h, w] 改变为 [h, w]
        # imgb = np.squeeze(imgb)

        # # imgb = Image.fromarray(np.int8(imgb))
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1,2,1),plt.xticks([]),plt.yticks([]),plt.imshow(imga)
        # plt.subplot(1,2,2),plt.xticks([]),plt.yticks([]),plt.imshow(imgb)
        # plt.show()

    @staticmethod
    def write_txt(txt, image_names, label_names=None):
        """
        write the image_names and label_names on the txt file like this:

        images/image_name labels/label_name
        ...

        or this when label is None.

        images/image_name
        ...

        """
        with open(txt, 'w') as f:
            for i in range(len(image_names)):
                if label_names is not None:
                    string = "{} {}\n".format('images/' + image_names[i],
                                              'labels/' + label_names[i])
                else:
                    string = "{}\n".format('images/' + image_names[i])

                f.write(string)

        print("successfully write to {}".format(txt))

    def split_files_txt(self, txt, image_files, label_files=None, split=None):
        """
        Split filenames and write the image names and label names on train.txt, val.txt or test.txt.
        Set the valset to 20% of images if all files need to be used in training.

        Args:
        txt(string): the path to the txt file, for example: "data/train.txt"
        image_files(list|tuple): the list of image names.
        label_files(list|tuple): the list of label names, order is corresponding with the image_files.
        split(float|int): Percentage of the dataset used in training

        """
        if split is None:
            if label_files is None:  # testset don't have
                split = len(image_files)
            else:
                split = int(0.8 * len(image_files))
        elif split <= 1:
            split = int(split * len(image_files))
        elif split > 1:
            raise RuntimeError(
                "Only have {} images but required {} images in trainset")

        if "train" in txt:
            image_names = image_files[:split]
            label_names = label_files[:split]
        elif "val" in txt:
            # set the valset to 20% of images if all files need to be used in training
            if split == len(image_files):
                valsplit = int(0.8 * len(image_files))
                image_names = image_files[valsplit:]
                label_names = label_files[valsplit:]
            else:
                image_names = image_files[split:]
                label_names = label_files[split:]
        elif "test" in txt:
            self.write_txt(txt, image_files[:split])

            return
        else:
            raise NotImplementedError(
                "The txt split except for train.txt, val.txt and test.txt is not implemented yet."
            )

        self.write_txt(txt, image_names, label_names)

    @staticmethod
    def set_image_infor(image_name, infor_dict):
        try:
            img_itk = sitk.ReadImage(image_name)
        except:
            add_qform_sform(image_name)
            img_itk = sitk.ReadImage(image_name)
        infor_dict["dim"] = img_itk.GetDimension()
        img_npy = sitk.GetArrayFromImage(img_itk)
        infor_dict["shape"] = [img_npy.shape, ]
        infor_dict["minmax_vals"] = [str(img_npy.min()), str(img_npy.max())]
        infor_dict["spacing"] = img_itk.GetSpacing()
        infor_dict["origin"] = img_itk.GetOrigin()
        infor_dict["direction"] = img_itk.GetDirection()

        return infor_dict

    def generate_dataset_json(self,
                              modalities,
                              labels,
                              dataset_name,
                              license_desc="hands off!",
                              dataset_description="",
                              dataset_reference="",
                              save_path=None):
        """
        :param save_path: This needs to be the full path to the dataset.json you intend to write, default is the raw_data_path
        :param images_dir: path to the images folder of that dataset
        :param labels_dir: path to the label folder of that dataset
        :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
        corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
        :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
        supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
        :param dataset_name: The name of the dataset. Can be anything you want
        :param license_desc:
        :param dataset_description:
        :param dataset_reference: website of the dataset, if available
        :return: saved dataset.json 
        """

        if save_path is not None:
            self.dataset_json_path = os.path.join(
                save_path, "dataset.json")  # save the dataset.json to raw path

        if osp.exists(self.dataset_json_path):
            print(
                f"Dataset json exists, skipping. Delete file {self.dataset_json_path} to regenerate."
            )
            return

        if not self.dataset_json_path.endswith("dataset.json"):
            print(
                "WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
                "Proceeding anyways...")

        json_dict = {}
        json_dict['name'] = dataset_name
        json_dict['description'] = dataset_description
        json_dict['reference'] = dataset_reference
        json_dict['licence'] = license_desc
        json_dict['modality'] = {
            str(i): modalities[i]
            for i in range(len(modalities))
        }
        json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

        # set information of training and testing file
        json_dict['training'] = {}

        for i, image_name in enumerate(
                tqdm(
                    self.image_files,
                    total=len(self.image_files),
                    desc="Load train file information into dataset.json")):
            infor_dict = {
                'image': image_name,
                "label": self.label_files[i]
            }  # nii.gz filename
            infor_dict = self.set_image_infor(image_name, infor_dict)
            json_dict['training'][image_name.split("/")[-1].split(".")[
                0]] = infor_dict

        json_dict['test'] = {}
        if self.image_files_test:
            for i, image_name in enumerate(
                    tqdm(
                        self.image_files_test,
                        total=len(self.image_files_test),
                        desc="Load Test file information")):
                infor_dict = {'image': image_name}
                infor_dict = self.set_image_infor(image_name, infor_dict)
                json_dict['test'][image_name.split("/")[-1].split(".")[
                    0]] = infor_dict

        with open(self.dataset_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)
            print("save dataset.json to {}".format(self.dataset_json_path))

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
import os
import shutil
import numpy as np
import SimpleITK as sitk
from multiprocessing import Pool

from .path_utils import join_paths


def convert_to_decathlon(input_folder,
                         output_folder='./',
                         data_json="dataset.json",
                         train_images_dir="imagesTr",
                         train_labels_dir="labelsTr",
                         test_images_dir="imagesTs",
                         num_processes=8):
    crawl_and_remove_hidden_from_decathlon(input_folder)
    split_4d(
        input_folder,
        output_folder=output_folder,
        data_json=data_json,
        train_images_dir=train_images_dir,
        train_labels_dir=train_labels_dir,
        test_images_dir=test_images_dir,
        num_processes=num_processes)


def crawl_and_remove_hidden_from_decathlon(folder,
                                           train_images_dir="imagesTr",
                                           train_labels_dir="labelsTr",
                                           test_images_Tr="imagesTs"):
    subf = [sub_dir for sub_dir in os.listdir(folder)]
    assert train_images_dir in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subdirs imagesTr, " \
                                                     "labelsTr and imagesTs."
    assert test_images_Tr in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subdirs imagesTr, " \
                                                     "labelsTr and imagesTs."
    assert train_labels_dir in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subdirs imagesTr, " \
                                                     "labelsTr and imagesTs."
    _ = [
        os.remove(join_paths(folder, sub_file))
        for sub_file in os.listdir(folder) if sub_file.startswith('.')
    ]
    _ = [
        os.remove(join_paths(folder, train_images_dir, sub_file))
        for sub_file in os.listdir(join_paths(folder, train_images_dir))
        if sub_file.startswith('.')
    ]
    _ = [
        os.remove(join_paths(folder, train_labels_dir, sub_file))
        for sub_file in os.listdir(join_paths(folder, train_labels_dir))
        if sub_file.startswith('.')
    ]
    _ = [
        os.remove(join_paths(folder, test_images_Tr, sub_file))
        for sub_file in os.listdir(join_paths(folder, test_images_Tr))
        if sub_file.startswith('.')
    ]


def split_4d_to_3d_and_save(img_itk: sitk.Image, output_folder, file_base_name):
    slicer = sitk.ExtractImageFilter()
    s = list(img_itk.GetSize())
    s[-1] = 0
    slicer.SetSize(s)
    for i, slice_idx in enumerate(range(img_itk.GetSize()[-1])):
        slicer.SetIndex([0, 0, 0, slice_idx])
        sitk_volume = slicer.Execute(img_itk)
        sitk.WriteImage(sitk_volume,
                        join_paths(output_folder,
                                   file_base_name[:-7] + "_%04.0d.nii.gz" % i))


def split_4d_nifti(filename, output_folder):
    img_itk = sitk.ReadImage(filename)
    dim = img_itk.GetDimension()
    file_base = filename.split("/")[-1]
    if dim == 3:
        shutil.copy(filename,
                    join_paths(output_folder, file_base[:-7] + "_0000.nii.gz"))
        return
    elif dim != 4:
        raise RuntimeError(
            "Unexpected dimensionality: {} of file {}, cannot split".format(
                dim, filename))
    else:
        split_4d_to_3d_and_save(img_itk, output_folder, file_base)


def split_4d(input_folder,
             output_folder='./',
             data_json="dataset.json",
             train_images_dir="imagesTr",
             train_labels_dir="labelsTr",
             test_images_dir="imagesTs",
             num_processes=8):
    assert os.path.isdir(join_paths(input_folder, train_images_dir)) and os.path.isdir(join_paths(input_folder, train_labels_dir)) and \
           os.path.isfile(join_paths(input_folder, data_json)), "The input folder must be a valid Task folder from the Medical Segmentation Decathlon with at least the " \
        "{} and {} subdirs and the {} file.".format(train_images_dir, train_labels_dir, test_images_dir)
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    shutil.copytree(
        join_paths(input_folder, train_labels_dir),
        join_paths(output_folder, train_labels_dir))
    shutil.copy(join_paths(input_folder, data_json), output_folder)

    files = []
    output_dirs = []
    os.makedirs(output_folder, exist_ok=True)
    for subdir in [train_images_dir, test_images_dir]:
        curr_out_dir = join_paths(output_folder, subdir)
        if not os.path.isdir(curr_out_dir):
            os.mkdir(curr_out_dir)
        curr_dir = join_paths(input_folder, subdir)
        nii_files = [
            join_paths(curr_dir, i) for i in os.listdir(curr_dir)
            if i.endswith(".nii.gz")
        ]
        nii_files.sort()
        for n in nii_files:
            files.append(n)
            output_dirs.append(curr_out_dir)

    p = Pool(num_processes)
    p.starmap(split_4d_nifti, zip(files, output_dirs))
    p.close()
    p.join()

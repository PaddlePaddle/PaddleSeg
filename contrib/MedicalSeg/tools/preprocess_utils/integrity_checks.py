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
import json
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from multiprocessing import Pool
from .load_image import load_series

from .path_utils import join_paths


def verify_all_same_orientation(folder):
    nii_files = [
        join_paths(folder, nii_path) for nii_path in os.listdir(folder)
        if os.path.isfile(join_paths(folder, nii_path)) and nii_path.endswith(
            ".nii.gz")
    ]
    orientations = []
    for n in nii_files:
        img = nib.load(n)
        affine = img.affine
        orientation = nib.aff2axcodes(affine)
        orientations.append(orientation)
    orientations = np.array(orientations)
    unique_orientations = np.unique(orientations, axis=0)
    all_same = len(unique_orientations) == 1
    return all_same, unique_orientations


def verify_same_geometry(img_1: sitk.Image, img_2: sitk.Image):
    ori1, spacing1, direction1, size1 = img_1.GetOrigin(), img_1.GetSpacing(
    ), img_1.GetDirection(), img_1.GetSize()
    ori2, spacing2, direction2, size2 = img_2.GetOrigin(), img_2.GetSpacing(
    ), img_2.GetDirection(), img_2.GetSize()
    return np.all(np.isclose(ori1, ori2)) and np.all(
        np.isclose(spacing1, spacing2)) and np.all(
            np.isclose(direction1, direction2)) and np.all(
                np.isclose(size1, size2))


def verify_contains_only_expected_labels(itk_img: str,
                                         valid_labels: (tuple, list)):
    img_npy, _, _ = load_series(itk_img)
    uniques = np.unique(img_npy)
    invalid_uniques = [i for i in uniques if i not in valid_labels]
    if len(invalid_uniques) == 0:
        r = True
    else:
        r = False
    return r, invalid_uniques


def verify_same_geometry_and_shape_and_nonan(image_paths, label_path):
    # verify that all modalities and the label have the same shape and geometry.
    label_itk = sitk.ReadImage(label_path)
    nans_in_seg = np.any(np.isnan(sitk.GetArrayFromImage(label_itk)))
    assert not nans_in_seg, "There are NAN values in label {}.".format(
        label_path)
    for image_path in image_paths:
        img = sitk.ReadImage(image_path)
        np_img = sitk.GetArrayFromImage(img)
        nans_in_image = np.any(np.isnan(np_img))
        assert not nans_in_image, "There are NAN values in image {}.".format(
            image_path)
        assert verify_same_geometry(img, label_itk), "The geometry of the image {} does not match the geometry of the label {}. The pixel arrays " \
                                "will not be aligned and nnU-Net cannot use this data. Please make sure your image modalities " \
                                    "are coregistered and have the same geometry as the label.".format(image_path, label_path)


def verify_training_dataset(folder,
                            num_modalities,
                            identifiers,
                            expected_labels,
                            train_images_dir="imagesTr",
                            train_labels_dir="labelsTr",
                            default_num_threads=8):
    imagesTr_folder = join_paths(folder, train_images_dir)
    nii_files_in_imagesTr = [
        nii_path for nii_path in os.listdir(imagesTr_folder)
        if os.path.isfile(join_paths(imagesTr_folder, nii_path)) and
        nii_path.endswith(".nii.gz")
    ]
    labelsTr_folder = join_paths(folder, train_labels_dir)
    nii_files_in_labelsTr = [
        nii_path for nii_path in os.listdir(labelsTr_folder)
        if os.path.isfile(join_paths(labelsTr_folder, nii_path)) and
        nii_path.endswith(".nii.gz")
    ]

    label_files = []
    for c in identifiers:
        # check if all files are present
        expected_label_file = join_paths(folder, train_labels_dir,
                                         c + ".nii.gz")
        label_files.append(expected_label_file)
        expected_image_files = [
            join_paths(folder, train_images_dir, c + "_%04.0d.nii.gz" % i)
            for i in range(num_modalities)
        ]
        assert os.path.isfile(
            expected_label_file
        ), "Could not find label file for case {}. Expected file: {}".format(
            c, expected_label_file)
        assert all(
            [os.path.isfile(i) for i in expected_image_files]
        ), "Some image files are missing for case {}. Expected files: {}.".format(
            c, expected_image_files)
        # check that all modalities and the label have the same shape and geometry
        verify_same_geometry_and_shape_and_nonan(expected_image_files,
                                                 expected_label_file)

        for i in expected_image_files:
            nii_files_in_imagesTr.remove(os.path.basename(i))
        nii_files_in_labelsTr.remove(os.path.basename(expected_label_file))

    assert len(
        nii_files_in_imagesTr
    ) == 0, "There are training cases in {} that are not listed in dataset json file.".format(
        train_images_dir)
    assert len(
        nii_files_in_labelsTr
    ) == 0, "There are training cases in {} that are not listed in dataset json file.".format(
        train_labels_dir)

    # check if labels are in consecutive order
    assert expected_labels[
        0] == 0, 'The first label must be 0 and maps to the background'
    labels_valid_consecutive = np.ediff1d(expected_labels) == 1
    assert all(
        labels_valid_consecutive
    ), f'Labels must be in consecutive order (0, 1, 2, ...). The labels {np.array(expected_labels)[1:][~labels_valid_consecutive]} do not satisfy this restriction'

    p = Pool(default_num_threads)
    results = p.starmap(verify_contains_only_expected_labels,
                        zip(label_files, [expected_labels] * len(label_files)))
    p.close()
    p.join()

    for i, r in enumerate(results):
        assert r[
            0], "Unexpected labels found in file {}. Found these unexpected values {}.".format(
                label_files[i], r[1])


def verify_test_dataset(folder,
                        num_modalities,
                        identifiers,
                        test_images_dir="imagesTs"):
    imagesTs_folder = join_paths(folder, test_images_dir)
    nii_files_in_imagesTs = [
        nii_path for nii_path in os.listdir(imagesTs_folder)
        if os.path.isfile(join_paths(imagesTs_folder, nii_path)) and
        nii_path.endswith(".nii.gz")
    ]
    for c in identifiers:
        # check if all files are present
        expected_image_files = [
            join_paths(folder, test_images_dir, c + "_%04.0d.nii.gz" % i)
            for i in range(num_modalities)
        ]
        assert all(
            [os.path.isfile(i) for i in expected_image_files]
        ), "Some image files are missing for case {}. Expected files: {}.".format(
            c, expected_image_files)

        # verify that all modalities have the same geometry. We use the affine for this
        if num_modalities > 1:
            images_itk = [sitk.ReadImage(i) for i in expected_image_files]
            reference_img = images_itk[0]
            for i, img in enumerate(images_itk[1:]):
                assert verify_same_geometry(img, reference_img), "The modalities of the image {} do not seem to be " \
                                                                    "registered. Please coregister your modalities.".foramt(
                                                                        expected_image_files[i])
        for i in expected_image_files:
            nii_files_in_imagesTs.remove(os.path.basename(i))
    assert len(
        nii_files_in_imagesTs
    ) == 0, "There are training cases in {} that are not listed in dataset json file. Their names are listed in {}.".format(
        test_images_dir, nii_files_in_imagesTs)


def verify_dataset_integrity(folder,
                             data_json="dataset.json",
                             train_images_dir="imagesTr",
                             train_labels_dir="labelsTr",
                             test_images_dir="imagesTs",
                             default_num_threads=8):
    assert os.path.isfile(
        join_paths(folder, data_json)
    ), "There needs to be a {} file in folder {}, but not found.".format(
        data_json, folder)
    assert os.path.isdir(
        join_paths(folder, train_images_dir)
    ), "There needs to be a {} subfolder in folder {}, but not found.".format(
        train_images_dir, folder)
    assert os.path.isdir(
        join_paths(folder, train_labels_dir)
    ), "There needs to be a {} subfolder in folder {}, but not found.".format(
        train_labels_dir, folder)

    with open(join_paths(folder, data_json), 'r') as f:
        dataset = json.load(f)
    training_cases = dataset['training']
    num_modalities = len(dataset['modality'].keys())
    test_cases = dataset['test']
    expected_train_identifiers = [
        i['image'].split("/")[-1].split('.')[0] for i in training_cases
    ]
    expected_test_identifiers = [
        i.split("/")[-1].split('.')[0] for i in test_cases
    ]
    expected_labels = list(int(i) for i in dataset['labels'].keys())

    # check training dataset orientation
    all_same, unique_orientations = verify_all_same_orientation(
        join_paths(folder, train_images_dir))
    assert all_same, "Not all images in the dataset have the same axis ordering. Please correct that by reorienting the data."

    # check duplicate label
    assert len(expected_train_identifiers) == len(
        np.unique(expected_train_identifiers)
    ), "Found duplicate training labels in {}, please check your dataset.".format(
        data_json)
    verify_training_dataset(
        folder,
        num_modalities,
        expected_train_identifiers,
        expected_labels,
        train_images_dir=train_images_dir,
        train_labels_dir=train_labels_dir,
        default_num_threads=default_num_threads)

    # check test set, but only if there actually is a test set
    if len(expected_test_identifiers) > 0:
        verify_test_dataset(
            folder,
            num_modalities,
            expected_test_identifiers,
            test_images_dir=test_images_dir)

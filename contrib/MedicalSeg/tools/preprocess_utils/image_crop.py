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
import shutil
import pickle
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes
from multiprocessing import Pool
from collections import OrderedDict

from .path_utils import join_paths


def crop(raw_data_dir,
         cropped_output_dir,
         data_json="dataset.json",
         train_images_dir="imagesTr",
         train_labels_dir="labelsTr",
         override=False,
         num_threads=8):
    os.makedirs(cropped_output_dir, exist_ok=True)
    if override and os.path.isdir(cropped_output_dir):
        shutil.rmtree(cropped_output_dir)
        os.makedirs(cropped_output_dir, exist_ok=True)

    lists, _ = create_lists_from_splitted_dataset(
        raw_data_dir,
        data_json=data_json,
        train_images_dir=train_images_dir,
        train_labels_dir=train_labels_dir)

    imgcrop = ImageCropper(cropped_output_dir, num_threads)
    imgcrop.run_cropping(lists, overwrite_existing=override)
    shutil.copy(join_paths(raw_data_dir, data_json), cropped_output_dir)


def create_lists_from_splitted_dataset(base_folder_splitted,
                                       data_json="dataset.json",
                                       train_images_dir="imagesTr",
                                       train_labels_dir="labelsTr"):
    lists = []
    json_file = join_paths(base_folder_splitted, data_json)
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities):
            cur_pat.append(
                join_paths(base_folder_splitted, train_images_dir, tr[
                    'image'].split("/")[-1].split('.')[0] + "_%04.0d.nii.gz" %
                           mod))
        cur_pat.append(
            join_paths(base_folder_splitted, train_labels_dir, tr['label']
                       .split("/")[-1]))
        lists.append(cur_pat)
    return lists, {int(i): d['modality'][str(i)] for i in d['modality'].keys()}


def create_nonzero_mask(data):
    assert len(data.shape) == 4 or len(
        data.shape
    ) == 3, "The input data must have shape (C, X, Y, Z) or shape (C, X, Y), but got {}.".format(
        data.shape)
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(
        image.shape
    ) == 3, "Function 'crop_to_bbox' only supports 3d images, but got image shape: {}.".format(
        image.shape)
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]),
               slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def get_case_identifier(case):
    case_identifier = case[0].split("/")[-1].split(".nii.gz")[0][:-5]
    return case_identifier


def get_case_identifier_from_npz(case):
    case_identifier = case.split("/")[-1][:-4]
    return case_identifier


def load_case_from_list_of_files(data_files, seg_file=None):
    assert isinstance(data_files, list) or isinstance(
        data_files, tuple
    ), "Single case must be either a list or a tuple including image path and label path, but got type {}.".format(
        type(data_files))
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(f) for f in data_files]
    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[
        [2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[
        [2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file
    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
    if seg_file is not None:
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
    else:
        seg_npy = None
    return data_npy.astype(np.float32), seg_npy, properties


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        for c in range(seg.shape[0]):
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    if seg is not None:
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    else:
        nonzero_mask = nonzero_mask.astype(int)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox


class ImageCropper:
    """
    This one finds a mask of nonzero elements (must be nonzero in all modalities) and crops the image to that mask.
    In the case of BRaTS and ISLES data this results in a significant reduction in image size.
    Args:
        num_threads (int): Number of threads. Default: None.
        output_folder (str, None): path to store the cropped data. Default: None.
    """

    def __init__(self, output_folder=None, num_threads=8):
        self.output_folder = output_folder
        self.num_threads = num_threads

        if self.output_folder is not None:
            print(self.output_folder)
            os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def crop(data, properties, seg=None):
        data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
        properties["crop_bbox"] = bbox
        properties['classes'] = np.unique(seg)
        seg[seg < -1] = 0
        properties["size_after_cropping"] = data[0].shape
        return data, seg, properties

    @staticmethod
    def crop_from_list_of_files(data_files, seg_file=None):
        data, seg, properties = load_case_from_list_of_files(data_files,
                                                             seg_file)
        return ImageCropper.crop(data, properties, seg)

    def load_crop_save(self, case, case_identifier, overwrite_existing=False):
        try:
            if overwrite_existing or (not os.path.isfile(
                    join_paths(self.output_folder, "%s.npz" % case_identifier)
            ) or not os.path.isfile(
                    join_paths(self.output_folder, "%s.pkl" % case_identifier))
                                      ):

                data, seg, properties = self.crop_from_list_of_files(case[:-1],
                                                                     case[-1])

                all_data = np.vstack((data, seg))
                np.savez_compressed(
                    join_paths(self.output_folder, "%s.npz" % case_identifier),
                    data=all_data)
                with open(
                        join_paths(self.output_folder,
                                   "%s.pkl" % case_identifier), 'wb') as f:
                    pickle.dump(properties, f)
        except Exception as e:
            raise UserWarning(
                "Exception occurs when crop {}, exception info: {}.".format(
                    case_identifier, e))

    def run_cropping(self,
                     list_of_files,
                     overwrite_existing=False,
                     output_folder=None):
        if output_folder is not None:
            self.output_folder = output_folder

        output_folder_gt = join_paths(self.output_folder, "gt_segmentations")
        os.makedirs(output_folder_gt, exist_ok=True)
        for j, case in enumerate(list_of_files):
            if case[-1] is not None:
                shutil.copy(case[-1], output_folder_gt)

        list_of_args = []
        for j, case in enumerate(list_of_files):
            case_identifier = get_case_identifier(case)
            list_of_args.append((case, case_identifier, overwrite_existing))

        p = Pool(self.num_threads)
        p.starmap(self.load_crop_save, list_of_args)
        p.close()
        p.join()

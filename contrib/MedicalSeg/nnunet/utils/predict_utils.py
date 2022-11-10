# Implementation of this model is borrowed and modified
# (from torch to paddle) from here:
# https://github.com/MIC-DKFZ/nnUNet

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

import os
import sys
import ast
import shutil
import json
import pickle
import numpy as np
from copy import deepcopy
from typing import Tuple, Union

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import SimpleITK as sitk
from multiprocessing import Process, Queue, Pool

from nnunet.utils import load_remove_save
from nnunet.transforms import resize_segmentation
from .utils import save_segmentation_nifti_from_softmax
from tools.preprocess_utils import get_do_separate_z, get_lowres_axis, resample_data_or_seg


def load_postprocessing(json_path):
    with open(json_path, 'rb') as f:
        a = json.load(f)
    if 'min_valid_object_sizes' in a.keys():
        min_valid_object_sizes = ast.literal_eval(a['min_valid_object_sizes'])
    else:
        min_valid_object_sizes = None
    return a['for_which_classes'], min_valid_object_sizes


def to_one_hot(seg, all_seg_labels=None):
    if all_seg_labels is None:
        all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
        result[i][seg == l] = 1
    return result


def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, output_files,
                             segs_from_prev_stage, classes, transpose_forward):
    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            print("preprocessing", output_file)
            d, _, dct = preprocess_fn(l)
            # print(output_file, dct)
            if segs_from_prev_stage[i] is not None:
                assert os.path.isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(
                    ".nii.gz"), "segs_from_prev_stage" \
                                " must point to a " \
                                "segmentation file"
                seg_prev = sitk.GetArrayFromImage(
                    sitk.ReadImage(segs_from_prev_stage[i]))
                # check to see if shapes match
                img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
                assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), "image and segmentation from previous " \
                                                                                 "stage don't have the same pixel array " \
                                                                                 "shape! image: %s, seg_prev: %s" % \
                                                                                 (l[0], segs_from_prev_stage[i])
                seg_prev = seg_prev.transpose(transpose_forward)
                seg_reshaped = resize_segmentation(
                    seg_prev, d.shape[1:], order=1)
                seg_reshaped = to_one_hot(seg_reshaped, classes)
                d = np.vstack((d, seg_reshaped)).astype(np.float32)
            """There is a problem with python process communication that prevents us from communicating objects 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            print(d.shape)
            if np.prod(d.shape) > (
                    2e9 / 4 * 0.85
            ):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            q.put((output_file, (d, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")


def preprocess_multithreaded(predictor,
                             list_of_lists,
                             output_files,
                             num_processes=2,
                             segs_from_prev_stage=None):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)

    num_processes = min(len(list_of_lists), num_processes)
    classes = list(range(1, predictor.num_classes))
    q = Queue(1)
    processes = []
    for i in range(num_processes):
        pr = Process(
            target=preprocess_save_to_queue,
            args=(predictor.preprocess_patient, q,
                  list_of_lists[i::num_processes],
                  output_files[i::num_processes],
                  segs_from_prev_stage[i::num_processes], classes,
                  predictor.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()
        q.close()


def predict_cases(predictor,
                  list_of_lists,
                  output_filenames,
                  save_npz,
                  num_threads_preprocessing,
                  num_threads_nifti_save,
                  segs_from_prev_stage=None,
                  do_tta=True,
                  mixed_precision=True,
                  overwrite_existing=False,
                  all_in_gpu=False,
                  step_size=0.5,
                  disable_postprocessing=False,
                  segmentation_export_kwargs: dict=None,
                  postprocessing_json_path=None):
    assert len(list_of_lists) == len(
        output_filenames
    ), "The number of input files and output files is not same. Please check your input folder."
    if segs_from_prev_stage is not None:
        assert len(segs_from_prev_stage) == len(
            output_filenames
        ), "The cascade lowres predict results is not the same with the number of output files."

    pool = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            os.makedirs(dr, exist_ok=True)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(os.path.join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        not_done_idx = [
            i for i, j in enumerate(cleaned_output_files)
            if (not os.path.isfile(j)
                ) or (save_npz and not os.path.isfile(j[:-7] + '.npz'))
        ]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [
                segs_from_prev_stage[i] for i in not_done_idx
            ]

        print("number of cases that still need to be predicted:",
              len(cleaned_output_files))

    if segmentation_export_kwargs is None:
        if 'segmentation_export_params' in predictor.plans.keys():
            force_separate_z = predictor.plans['segmentation_export_params'][
                'force_separate_z']
            interpolation_order = predictor.plans['segmentation_export_params'][
                'interpolation_order']
            interpolation_order_z = predictor.plans[
                'segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
    else:
        force_separate_z = segmentation_export_kwargs['force_separate_z']
        interpolation_order = segmentation_export_kwargs['interpolation_order']
        interpolation_order_z = segmentation_export_kwargs[
            'interpolation_order_z']

    print("starting preprocessing generator")
    print("file list: ", list_of_lists)
    print("preprocess output files: ", cleaned_output_files)
    preprocessing = preprocess_multithreaded(
        predictor, list_of_lists, cleaned_output_files,
        num_threads_preprocessing, segs_from_prev_stage)
    print("starting prediction...")
    all_output_files = []
    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed
        all_output_files.append(all_output_files)
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data

        print("predicting", output_filename)
        softmax = predictor.multi_folds_predict_preprocessed_data_return_seg_and_softmax(
            data=d,
            do_mirroring=do_tta,
            mirror_axes=predictor.data_aug_params['mirror_axes'],
            use_sliding_window=True,
            step_size=step_size,
            use_gaussian=True,
            mixed_precision=mixed_precision, )

        transpose_forward = predictor.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = predictor.plans.get('transpose_backward')
            softmax = softmax.transpose([0] +
                                        [i + 1 for i in transpose_backward])

        if save_npz:
            npz_file = output_filename[:-7] + ".npz"
        else:
            npz_file = None

        if hasattr(predictor, 'regions_class_order'):
            region_class_order = predictor.regions_class_order
        else:
            region_class_order = None

        bytes_per_voxel = 4
        if all_in_gpu:
            bytes_per_voxel = 2
        if np.prod(softmax.shape) > (2e9 / bytes_per_voxel * 0.85):
            print(
                "This output is too large for python process-process communication. Saving output temporarily to disk"
            )
            np.save(output_filename[:-7] + ".npy", softmax)
            softmax = output_filename[:-7] + ".npy"

        results.append(
            pool.starmap_async(save_segmentation_nifti_from_softmax, (
                (softmax, output_filename, dct, interpolation_order,
                 region_class_order, None, None, npz_file, None,
                 force_separate_z, interpolation_order_z), )))

    print(
        "inference done. Now waiting for the segmentation export to finish...")
    _ = [i.get() for i in results]

    if not disable_postprocessing:
        results = []
        pp_file = postprocessing_json_path
        if os.path.isfile(pp_file):
            print("postprocessing...")
            shutil.copy(pp_file,
                        os.path.abspath(os.path.dirname(output_filenames[0])))
            for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
            results.append(
                pool.starmap_async(
                    load_remove_save,
                    zip(output_filenames, output_filenames,
                        [for_which_classes] * len(output_filenames),
                        [min_valid_obj_size] * len(output_filenames))))
            _ = [i.get() for i in results]
        else:
            print(
                "WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
                "consolidate_folds in the output folder of the model first!")

    pool.close()
    pool.join()

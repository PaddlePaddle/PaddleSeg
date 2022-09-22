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
import numpy as np

import paddle

from typing import Tuple
from multiprocessing import Pool
from . import resample_and_save, DynamicPredictor


def predict_preprocessed_data_return_seg_and_softmax(
        predictor,
        dataset,
        data: np.ndarray,
        do_mirroring: bool=True,
        mirror_axes: Tuple[int]=None,
        use_sliding_window: bool=True,
        step_size: float=0.5,
        use_gaussian: bool=True,
        pad_border_mode: str='constant',
        pad_kwargs: dict=None,
        verbose: bool=True,
        mixed_precision: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    if pad_border_mode == 'constant' and pad_kwargs is None:
        pad_kwargs = {'constant_values': 0}

    if do_mirroring and mirror_axes is None:
        mirror_axes = dataset.data_aug_params['mirror_axes']

    if do_mirroring:
        assert dataset.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                  "was done without mirroring"
    with paddle.no_grad():
        argmax_pred, softmax_pred = predictor.predict_3D(
            data,
            do_mirroring=do_mirroring,
            mirror_axes=mirror_axes,
            use_sliding_window=use_sliding_window,
            step_size=step_size,
            patch_size=dataset.patch_size,
            regions_class_order=None,
            use_gaussian=use_gaussian,
            pad_border_mode=pad_border_mode,
            pad_kwargs=pad_kwargs,
            verbose=verbose,
            mixed_precision=mixed_precision)
    return argmax_pred, softmax_pred


def predict_next_stage(model,
                       plans,
                       dataset,
                       output_folder,
                       stage_to_be_predicted_folder,
                       mixed_precision,
                       num_threads=4):
    output_folder = os.path.join(output_folder, "pred_next_stage")
    os.makedirs(output_folder, exist_ok=True)

    if 'segmentation_export_params' in plans.keys():
        force_separate_z = plans['segmentation_export_params'][
            'force_separate_z']
        interpolation_order = plans['segmentation_export_params'][
            'interpolation_order']
        interpolation_order_z = plans['segmentation_export_params'][
            'interpolation_order_z']
    else:
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0

    export_pool = Pool(num_threads)
    results = []

    model.eval()
    predictor = DynamicPredictor(model)

    for pat in dataset.dataset_val.keys():
        print(pat, 'predict next stage...')
        data_file = dataset.dataset_val[pat]['data_file']
        data_preprocessed = np.load(data_file)['data'][:-1]
        data_file_nofolder = data_file.split("/")[-1]
        data_file_nextstage = os.path.join(stage_to_be_predicted_folder,
                                           data_file_nofolder)
        data_nextstage = np.load(data_file_nextstage)['data']
        target_shp = data_nextstage.shape[1:]
        output_file = os.path.join(
            output_folder,
            data_file_nextstage.split("/")[-1][:-4] + "_segFromPrevStage.npz")
        if os.path.exists(output_file):
            print("{} already exists, skip.".format(output_file))
            continue

        _, predicted_probabilities = predict_preprocessed_data_return_seg_and_softmax(
            predictor,
            dataset,
            data_preprocessed,
            do_mirroring=dataset.data_aug_params["do_mirror"],
            mirror_axes=dataset.data_aug_params['mirror_axes'],
            mixed_precision=mixed_precision)

        if np.prod(predicted_probabilities.shape) > (2e9 / 4 * 0.85):
            np.save(output_file[:-4] + ".npy", predicted_probabilities)
            predicted_probabilities = output_file[:-4] + ".npy"

        results.append(
            export_pool.starmap_async(resample_and_save, [(
                predicted_probabilities, target_shp, output_file,
                force_separate_z, interpolation_order, interpolation_order_z)]))

    _ = [i.get() for i in results]
    export_pool.close()
    export_pool.join()

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

import argparse
import os
import sys
import pickle
import shutil
import numpy as np
from copy import deepcopy
from typing import Tuple, List, Union

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import paddle
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

from nnunet.utils.static_predictor import StaticPredictor
from nnunet.transforms import default_2D_augmentation_params, default_3D_augmentation_params
from nnunet.predict import predict_from_folder
from tools.preprocess_utils import GenericPreprocessor, PreprocessorFor2D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        help="Must contain all modalities for each patient in the correct"
        " order (same as training). Files must be named "
        "CASENAME_XXXX.nii.gz where XXXX is the modality "
        "identifier (0000, 0001, etc)",
        required=True)
    parser.add_argument(
        '--output_folder', required=True, help="folder for saving predictions")
    parser.add_argument(
        '--model_type',
        required=True,
        type=str,
        help="Model type, only support '2d', '3d', 'cascade_lowres', 'cascade_fullres'."
    )

    parser.add_argument(
        '--plan_path', required=True, type=str, help='the path to plan_path')
    parser.add_argument(
        '--model_paths',
        nargs='+',
        required=True,
        help="The multi pdmodel paths.")
    parser.add_argument(
        '--param_paths',
        nargs='+',
        required=True,
        help="The multi pdiparams paths.")
    parser.add_argument(
        '--postprocessing_json_path',
        required=True,
        default=None,
        type=str,
        help='the path to postprocessing json.')
    parser.add_argument(
        '--folds',
        required=False,
        type=int,
        default=5,
        help='number of folds, default: 5.')
    parser.add_argument(
        '--lowres_segmentations',
        required=False,
        default=None,
        help="If model is the highres stage of the cascade then you can use this folder to provide "
        "predictions from the low resolution 3D U-Net.")
    parser.add_argument(
        '--save_npz',
        required=False,
        action='store_true',
        help="use this if you want to ensemble these predictions with those of other models. Softmax "
        "probabilities will be saved as compressed numpy arrays in output_folder and can be "
        "merged between output_folders with nnUNet_ensemble_predictions")

    parser.add_argument(
        "--num_threads_preprocessing",
        required=False,
        default=6,
        type=int,
        help="Determines many background processes will be used for data preprocessing. Reduce this if you "
        "run into out of memory (RAM) problems. Default: 6")

    parser.add_argument(
        "--num_threads_nifti_save",
        required=False,
        default=2,
        type=int,
        help="Determines many background processes will be used for segmentation export. Reduce this if you "
        "run into out of memory (RAM) problems. Default: 2")

    parser.add_argument(
        "--mode", type=str, default="normal", required=False, help="Hands off!")
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.5,
        required=False,
        help="don't touch")
    parser.add_argument(
        "--overwrite_existing",
        required=False,
        default=False,
        action="store_true",
        help="Set this flag if the target folder contains predictions that you would like to overwrite"
    )
    parser.add_argument(
        "--disable_postprocessing",
        required=False,
        default=False,
        action="store_true",
        help="Set this flag if no need postprocessing")
    parser.add_argument(
        "--disable_tta",
        required=False,
        default=False,
        action="store_true",
        help="set this flag to disable test time data augmentation via mirroring. Speeds up inference "
        "by roughly factor 4 (2D) or 8 (3D)")
    parser.add_argument(
        '--min_subgraph_size',
        default=3,
        type=int,
        help='The min subgraph size in tensorrt prediction.')
    return parser.parse_args()


class StaticMultiFolderPredictor:
    def __init__(self,
                 model_paths,
                 param_paths,
                 plan_path,
                 stage,
                 min_subgraph_size=3):
        self.stage = stage
        self.plans = self.load_plans(plan_path)
        self.num_classes = self.plans['num_classes'] + 1
        self.patch_size = np.array(self.plans['plans_per_stage'][self.stage][
            'patch_size']).astype(int)
        if len(self.patch_size) == 2:
            self.threeD = False
            self.data_aug_params = default_2D_augmentation_params
        elif len(self.patch_size) == 3:
            self.threeD = True
            self.data_aug_params = default_3D_augmentation_params
        self.intensity_properties = self.plans['dataset_properties'][
            'intensityproperties']
        self.normalization_schemes = self.plans['normalization_schemes']
        self.use_mask_for_norm = self.plans['use_mask_for_norm']
        if self.plans.get('transpose_forward') is None or self.plans.get(
                'transpose_backward') is None:
            print(
                "WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!"
            )
            self.plans['transpose_forward'] = [0, 1, 2]
            self.plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = self.plans['transpose_forward']
        self.transpose_backward = self.plans['transpose_backward']

        self.predictors = []
        for model_path, param_path in zip(model_paths, param_paths):
            self.predictors.append(
                StaticPredictor(model_path, param_path, self.plans, stage,
                                min_subgraph_size))

    def load_plans(self, plan_path):
        with open(plan_path, 'rb') as f:
            plans = pickle.load(f)
        return plans

    def preprocess_patient(self, input_files):
        if self.threeD:
            preprocessor_class = GenericPreprocessor
        else:
            preprocessor_class = PreprocessorFor2D

        preprocessor = preprocessor_class(
            self.normalization_schemes, self.use_mask_for_norm,
            self.transpose_forward, self.intensity_properties)
        d, s, properties = preprocessor.preprocess_test_case(
            input_files,
            self.plans['plans_per_stage'][self.stage]['current_spacing'])
        return d, s, properties

    def multi_folds_predict_preprocessed_data_return_seg_and_softmax(
            self,
            data: np.ndarray,
            do_mirroring: bool=True,
            mirror_axes: Tuple[int]=None,
            use_sliding_window: bool=True,
            step_size: float=0.5,
            use_gaussian: bool=True,
            pad_border_mode: str='constant',
            pad_kwargs: dict=None,
            verbose: bool=True,
            mixed_precision=True):
        softmax_res = None
        for predictor in self.predictors:
            x = predictor.predict_preprocessed_data_return_seg_and_softmax(
                data=data,
                do_mirroring=do_mirroring,
                mirror_axes=mirror_axes,
                use_sliding_window=use_sliding_window,
                step_size=step_size,
                use_gaussian=use_gaussian,
                pad_border_mode=pad_border_mode,
                pad_kwargs=pad_kwargs,
                verbose=verbose,
                mixed_precision=mixed_precision)[1]
            if softmax_res is None:
                softmax_res = x
            else:
                softmax_res += x
        return softmax_res / len(self.predictors)


def main(args):
    assert args.model_type in [
        '2d', '3d', 'cascade_lowres', 'cascade_fullres'
    ], "model only support ['2d', '3d', 'cascade_lowres', 'cascade_fullres'], but got {}.".format(
        args.model_type)
    assert len(args.model_paths) == len(
        args.param_paths
    ), "The number of pdmodel is not the same with pdiparams. {} != {}.".format(
        len(args.model_paths), len(args.param_paths))
    print("model type: ", args.model_type)
    print("The plan path: ", args.plan_path)
    print("The model paths: ", args.model_paths)
    print("The postprocessing json path: ", args.postprocessing_json_path)

    if args.model_type in ['3d', 'cascade_fullres']:
        stage = 1
    else:
        stage = 0

    predictor = StaticMultiFolderPredictor(args.model_paths, args.param_paths,
                                           args.plan_path, stage,
                                           args.min_subgraph_size)

    if args.lowres_segmentations is not None:
        assert args.model_type == 'cascade_fullres', "You supply lowres_segmentations dir but the model is not 'cascade_fullres'. Please check model_type."
        print("Cascade lowres segmentation result dir: ",
              args.lowres_segmentations)

    predict_from_folder(
        predictor=predictor,
        input_folder=args.image_folder,
        output_folder=args.output_folder,
        save_npz=args.save_npz,
        num_threads_preprocessing=args.num_threads_preprocessing,
        num_threads_nifti_save=args.num_threads_nifti_save,
        lowres_segmentations=args.lowres_segmentations,
        tta=not args.disable_tta,
        mixed_precision=False,
        overwrite_existing=args.overwrite_existing,
        mode='normal',
        step_size=args.step_size,
        plan_path=args.plan_path,
        disable_postprocessing=args.disable_postprocessing,
        postprocessing_json_path=args.postprocessing_json_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)

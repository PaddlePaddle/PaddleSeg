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

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from medicalseg.models import NNUNet
from nnunet.utils import predict_cases, MultiFoldsPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        '--image_folder',
        help="Must contain all modalities for each patient in the correct"
        " order (same as training). Files must be named "
        "CASENAME_XXXX.nii.gz where XXXX is the modality "
        "identifier (0000, 0001, etc)",
        required=True)
    parser.add_argument(
        '--output_folder',
        "--output_folder",
        required=True,
        help="folder for saving predictions")
    parser.add_argument(
        '--model_type',
        '--model_type',
        required=True,
        type=str,
        help="Model type, only support '2d', '3d', 'cascade_lowres', 'cascade_fullres'."
    )

    parser.add_argument(
        '--plan_path',
        "--plan_path",
        required=True,
        type=str,
        help='the path to plan_path')
    parser.add_argument(
        '--model_paths',
        '--model_paths',
        nargs='+',
        required=True,
        help="The multi model paths.")
    parser.add_argument(
        '--postprocessing_json_path',
        "--postprocessing_json_path",
        required=True,
        default=None,
        type=str,
        help='the path to postprocessing json.')
    parser.add_argument(
        '--folds',
        "--folds",
        required=False,
        type=int,
        default=5,
        help='number of folds, default: 5.')
    parser.add_argument(
        '--lowres_segmentations',
        '--lowres_segmentations',
        required=False,
        default=None,
        help="If model is the highres stage of the cascade then you can use this folder to provide "
        "predictions from the low resolution 3D U-Net.")
    parser.add_argument(
        '--save_npz',
        '--save_npz',
        required=False,
        action='store_true',
        help="use this if you want to ensemble these predictions with those of other models. Softmax "
        "probabilities will be saved as compressed numpy arrays in output_folder and can be "
        "merged between output_folders with nnUNet_ensemble_predictions")

    parser.add_argument(
        "--num_threads_preprocessing",
        "--num_threads_preprocessing",
        required=False,
        default=6,
        type=int,
        help="Determines many background processes will be used for data preprocessing. Reduce this if you "
        "run into out of memory (RAM) problems. Default: 6")

    parser.add_argument(
        "--num_threads_nifti_save",
        "--num_threads_nifti_save",
        required=False,
        default=2,
        type=int,
        help="Determines many background processes will be used for segmentation export. Reduce this if you "
        "run into out of memory (RAM) problems. Default: 2")

    parser.add_argument(
        "--mode",
        "--mode",
        type=str,
        default="normal",
        required=False,
        help="Hands off!")
    parser.add_argument(
        "--step_size",
        "--step_size",
        type=float,
        default=0.5,
        required=False,
        help="don't touch")
    parser.add_argument(
        "--overwrite_existing",
        "--overwrite_existing",
        required=False,
        default=False,
        action="store_true",
        help="Set this flag if the target folder contains predictions that you would like to overwrite"
    )
    parser.add_argument(
        "--disable_postprocessing",
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
        '--precision',
        default='fp16',
        required=False,
        help="whether use mixed precision prediction.")
    return parser.parse_args()


def check_input_folder_and_return_caseIDs(input_folder,
                                          expected_num_modalities):
    print("This model expects {} input modalities for each image.".format(
        expected_num_modalities))
    files = [
        file for file in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, file)) and file.endswith(
            ".nii.gz")
    ]
    files.sort()
    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []
    assert len(
        files
    ) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not os.path.isfile(
                    os.path.join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)
    print("Found {} unique case ids, here are some examples: ".format(
        len(maybe_case_ids)),
          np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print(
        "If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc"
    )

    if len(remaining) > 0:
        print(
            "found {} unexpected remaining files in the folder. Here are some examples:".
            format(len(remaining)),
            np.random.choice(remaining, min(len(remaining), 10)))
    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")
    return maybe_case_ids


def predict_from_folder(predictor,
                        input_folder: str,
                        output_folder: str,
                        save_npz: bool,
                        num_threads_preprocessing: int,
                        num_threads_nifti_save: int,
                        lowres_segmentations,
                        tta: bool,
                        mixed_precision: bool=True,
                        overwrite_existing: bool=True,
                        mode: str='normal',
                        step_size: float=0.5,
                        segmentation_export_kwargs: dict=None,
                        disable_postprocessing: bool=False,
                        plan_path=None,
                        postprocessing_json_path=None):
    os.makedirs(output_folder, exist_ok=True)

    assert os.path.exists(plan_path), "plan file path: {} not exists.".format(
        plan_path)
    shutil.copy(plan_path, output_folder)

    with open(plan_path, 'rb') as f:
        expected_num_modalities = pickle.load(f)['num_modalities']
    case_ids = check_input_folder_and_return_caseIDs(input_folder,
                                                     expected_num_modalities)
    output_files = [
        os.path.join(output_folder, i + ".nii.gz") for i in case_ids
    ]
    all_files = [
        file for file in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, file)) and file.endswith(
            ".nii.gz")
    ]
    all_files.sort()
    list_of_lists = [[
        os.path.join(input_folder, i) for i in all_files
        if i[:len(j)].startswith(j) and len(i) == (len(j) + 12)
    ] for j in case_ids]

    if lowres_segmentations is not None:
        assert os.path.isdir(
            lowres_segmentations
        ), "if lowres_segmentations is not None then it must point to a directory"
        lowres_segmentations = [
            os.path.join(lowres_segmentations, i + ".nii.gz") for i in case_ids
        ]
        assert all(
            [os.path.isfile(i) for i in lowres_segmentations]), "not all lowres_segmentations files are present. " \
                                                                "(I was searching for case_id.nii.gz in that folder)"
    else:
        lowres_segmentations = None

    if mode == "normal":
        return predict_cases(
            predictor=predictor,
            list_of_lists=list_of_lists,
            output_filenames=output_files,
            save_npz=save_npz,
            num_threads_preprocessing=num_threads_preprocessing,
            num_threads_nifti_save=num_threads_nifti_save,
            segs_from_prev_stage=lowres_segmentations,
            do_tta=tta,
            mixed_precision=mixed_precision,
            overwrite_existing=overwrite_existing,
            step_size=step_size,
            segmentation_export_kwargs=segmentation_export_kwargs,
            disable_postprocessing=disable_postprocessing,
            postprocessing_json_path=postprocessing_json_path)


def main(args):
    assert args.model_type in [
        '2d', '3d', 'cascade_lowres', 'cascade_fullres'
    ], "model only support ['2d', '3d', 'cascade_lowres', 'cascade_fullres'], but got {}.".format(
        args.model_type)
    print("model type: ", args.model_type)
    print("The plan path: ", args.plan_path)
    print("The model paths: ", args.model_paths)
    print("The postprocessing json path: ", args.postprocessing_json_path)

    if args.model_type in ['3d', 'cascade_fullres']:
        stage = 1
    else:
        stage = 0
    if args.model_type in ['cascade_fullres', 'cascade_fullres']:
        cascade = True
    else:
        cascade = False

    model = NNUNet(plan_path=args.plan_path, stage=stage, cascade=cascade)
    predictor = MultiFoldsPredictor(model, args.model_paths)

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
        mixed_precision=args.precision == 'fp16',
        overwrite_existing=args.overwrite_existing,
        mode='normal',
        step_size=args.step_size,
        plan_path=args.plan_path,
        disable_postprocessing=args.disable_postprocessing,
        postprocessing_json_path=args.postprocessing_json_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)

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
import json
import shutil

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from typing import Tuple
from nnunet.utils import aggregate_scores, determine_postprocessing


def collect_cv_niftis(cv_folder: str,
                      output_folder: str,
                      validation_folder_name: str='validation_raw',
                      folds: tuple=(0, 1, 2, 3, 4)):
    validation_raw_folders = [
        os.path.join(cv_folder, "fold_%d" % i, validation_folder_name)
        for i in folds
    ]
    exist = [os.path.isdir(i) for i in validation_raw_folders]

    if not all(exist):
        raise RuntimeError(
            "some folds are missing. Please run the full 5-fold cross-validation. "
            "The following folds seem to be missing: %s" %
            [i for j, i in enumerate(folds) if not exist[j]])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    for f in folds:
        folder = validation_raw_folders[f]
        niftis = [
            os.path.join(folder, i) for i in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, i)) and i.endswith(".nii.gz")
        ]
        for n in niftis:
            shutil.copy(n, output_folder)


def consolidate_folds(output_folder_base,
                      validation_folder_name: str='validation_raw',
                      advanced_postprocessing: bool=False,
                      folds: Tuple[int]=(0, 1, 2, 3, 4),
                      num_threads=4):
    output_folder_raw = os.path.join(output_folder_base, "cv_niftis_raw")
    if os.path.isdir(output_folder_raw):
        shutil.rmtree(output_folder_raw)

    output_folder_gt = os.path.join(output_folder_base, "gt_niftis")
    collect_cv_niftis(output_folder_base, output_folder_raw,
                      validation_folder_name, folds)

    num_niftis_gt = len(
        subfiles(
            os.path.join(output_folder_base, "gt_niftis"), suffix='.nii.gz'))
    num_niftis = len(subfiles(output_folder_raw, suffix='.nii.gz'))
    if num_niftis != num_niftis_gt:
        raise AssertionError(
            "If does not seem like you trained all the folds! Train all folds first!"
        )

    with open(
            os.path.join(output_folder_base, "fold_0", validation_folder_name,
                         "summary.json"), 'r') as f:
        json_data = json.load(f)
    summary_info = json_data['results']['mean']

    classes = [int(i) for i in summary_info.keys()]
    niftis = [
        file for file in os.listdir(output_folder_raw)
        if os.path.isfile(os.path.join(output_folder_raw, file)) and
        file.endswith('.nii.gz')
    ]
    test_pred_pairs = [
        (os.path.join(output_folder_raw, i), os.path.join(output_folder_gt, i))
        for i in niftis
    ]

    _ = aggregate_scores(
        test_pred_pairs,
        labels=classes,
        json_output_file=os.path.join(output_folder_raw, "summary.json"),
        num_threads=num_threads)

    determine_postprocessing(
        output_folder_base,
        output_folder_gt,
        'cv_niftis_raw',
        final_subf_name="cv_niftis_postprocessed",
        processes=num_threads,
        advanced_postprocessing=advanced_postprocessing)


def subfiles(folder: str, suffix: str=None):
    res = [
        os.path.join(folder, i) for i in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, i)) and (suffix is None or
                                                        i.endswith(suffix))
    ]
    return res


def parse_args():
    parser = argparse.ArgumentParser(description='NNUnet model evaluation')

    parser.add_argument(
        '--gt_dir',
        dest='gt_dir',
        help='The path to gt result',
        type=str,
        default=None)
    parser.add_argument(
        '--val_pred_dir',
        dest='val_pred_dir',
        help='The path to val data result',
        type=str,
        default=None)
    parser.add_argument(
        '--folds',
        dest='folds',
        help='The number of cross validation folds.',
        type=int,
        default=5)
    return parser.parse_args()


def main(args):
    output_folder = args.val_pred_dir
    folds = [i for i in range(args.folds)]

    postprocessing_json = os.path.join(output_folder, "postprocessing.json")
    cv_niftis_folder = os.path.join(output_folder, "cv_niftis_raw")

    if not os.path.isfile(postprocessing_json) or not os.path.isdir(
            cv_niftis_folder):
        print("running missing postprocessing.")
        consolidate_folds(output_folder, folds=folds)
        assert os.path.isfile(
            postprocessing_json
        ), "Postprocessing json missing, expected: %s" % postprocessing_json
        assert os.path.isdir(
            cv_niftis_folder
        ), "Folder with niftis from CV missing, expected: %s" % cv_niftis_folder


if __name__ == '__main__':
    args = parse_args()
    main(args)

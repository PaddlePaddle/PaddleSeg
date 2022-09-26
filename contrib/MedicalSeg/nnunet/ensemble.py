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
import ast
import json
import numpy as np
import pickle
import shutil
from copy import deepcopy
from multiprocessing import Pool
from itertools import combinations

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from nnunet.utils import save_segmentation_nifti_from_softmax, load_remove_save, aggregate_scores, determine_postprocessing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ensemble_folds',
        '--ensemble_folds',
        nargs='+',
        required=True,
        help="The dirs of softmax results.")
    parser.add_argument(
        '--output_folder',
        '--output_folder',
        required=True,
        default='output/ensemble',
        help="The dirs of softmax results.")
    parser.add_argument(
        '--num_threads',
        '--num_threads',
        type=int,
        default=2,
        help='The number of threads.')
    parser.add_argument(
        "--gt_dir",
        "--gt_dir",
        required=False,
        default=None,
        type=str,
        help='grount truth dir')
    parser.add_argument(
        "--plan_path",
        "--plan_path",
        required=False,
        default=None,
        type=str,
        help='If ensemble val folder, plan_path must be supplied.')
    parser.add_argument(
        "--folds", "--folds", default=5, type=int, required=False)
    parser.add_argument(
        '--postprocessing_json_path',
        "--postprocessing_json_path",
        required=False,
        default=None,
        type=str,
        help='the path to postprocessing json.')
    parser.add_argument(
        '--save_npz',
        action="store_true",
        required=False,
        help="stores npz and pkl")
    return parser.parse_args()


def load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def merge_files(files,
                properties_files,
                out_file,
                override=False,
                store_npz=False):
    if override or not os.path.exists(out_file):
        print("ensemble files: ", files)
        softmax = [np.load(file)['softmax'][None] for file in files]
        softmax = np.vstack(softmax)
        softmax = np.mean(softmax, 0)
        props = [load_pickle(f) for f in properties_files]
        reg_class_orders = [
            p['regions_class_order']
            if 'regions_class_order' in p.keys() else None for p in props
        ]
        if not all([i is None for i in reg_class_orders]):
            tmp = reg_class_orders[0]
            for r in reg_class_orders[1:]:
                assert tmp == r, 'If merging files with regions_class_order, the regions_class_orders of all files must be the same. regions_class_order: {}, \n files: {}.'.format(
                    str(reg_class_orders), str(files))
            regions_class_order = tmp
        else:
            regions_class_order = None
        save_segmentation_nifti_from_softmax(
            softmax,
            out_file,
            props[0],
            3,
            regions_class_order,
            None,
            None,
            force_separate_z=None)
        if store_npz:
            np.savez_compressed(out_file[:-7] + ".npz", softmax=softmax)
            with open(out_file[:-7] + ".pkl", 'wb') as f:
                pickle.dump(props, f)
    else:
        print("{} already exists. Skip.".format(out_file))


def merge(folders,
          output_folder,
          threads,
          override=True,
          postprocessing_file=None,
          store_npz=False):
    print('ensemble predict...')
    if postprocessing_file is not None:
        output_folder_post = os.path.join(output_folder, 'postprocessed')
        os.makedirs(output_folder_post, exist_ok=True)
    else:
        output_folder_post = None
    output_folder = os.path.join(output_folder, 'no_postprocessed')
    os.makedirs(output_folder, exist_ok=True)

    patient_ids = []
    for folder in folders:
        patients = []
        for file in os.listdir(folder):
            if file.endswith(".npz"):
                patients.append(file)
        patient_ids.append(patients)
    patient_ids = [i for j in patient_ids for i in j]
    patient_ids = [i[:-4] for i in patient_ids]
    patient_ids = np.unique(patient_ids)

    for f in folders:
        print([os.path.join(f, i + ".npz") for i in patient_ids])
        assert all([os.path.isfile(os.path.join(f, i + ".npz")) for i in patient_ids]), "Not all patient npz are available in " \
                                                                        "all folders"
        assert all([os.path.isfile(os.path.join(f, i + ".pkl")) for i in patient_ids]), "Not all patient pkl are available in " \
                                                                        "all folders"

    files = []
    property_files = []
    out_files = []
    for p in patient_ids:
        files.append([os.path.join(f, p + ".npz") for f in folders])
        property_files.append([os.path.join(f, p + ".pkl") for f in folders])
        out_files.append(os.path.join(output_folder, p + ".nii.gz"))

    p = Pool(threads)
    p.starmap(merge_files,
              zip(files, property_files, out_files, [override] * len(out_files),
                  [store_npz] * len(out_files)))
    p.close()
    p.join()

    if postprocessing_file is not None:
        with open(postprocessing_file, 'rb') as f:
            pp_info = json.load(f)
        if 'min_valid_object_sizes' in pp_info.keys():
            min_valid_object_sizes = ast.literal_eval(pp_info[
                'min_valid_object_sizes'])
        else:
            min_valid_object_sizes = None
        for_which_classes, min_valid_obj_size = pp_info[
            'for_which_classes'], min_valid_object_sizes
        print('Postprocessing...')
        p = Pool(threads)
        nii_files = [
            file for file in os.listdir(output_folder)
            if file.endswith('.nii.gz')
        ]
        input_files = [os.path.join(output_folder, i) for i in nii_files]
        out_files = [os.path.join(output_folder_post, i) for i in nii_files]
        results = p.starmap_async(load_remove_save,
                                  zip(input_files, out_files,
                                      [for_which_classes] * len(input_files),
                                      [min_valid_obj_size] * len(input_files)))
        res = results.get()
        p.close()
        p.join()

        shutil.copy(postprocessing_file, output_folder_post)
        print('Postprocessing Done!')
    print("Ensemble predict Done!")


def ensemble_val(val_folder1,
                 val_folder2,
                 output_folder,
                 plan_path,
                 validation_folder,
                 folds,
                 gt_dir,
                 threads=4,
                 override=True):
    print("\nEnsembling folders: \n", val_folder1, "\n", val_folder2)

    output_folder_base = output_folder
    output_folder = os.path.join(output_folder_base, "ensembled_raw")

    with open(plan_path, 'rb') as f:
        plans = pickle.load(f)

    files = []
    property_files = []
    out_files = []
    gt_segmentations = []

    folder_with_gt_segs = gt_dir

    for f in range(folds):
        validation_folder_net1 = os.path.join(val_folder1, "fold_%d" % f,
                                              validation_folder)
        validation_folder_net2 = os.path.join(val_folder2, "fold_%d" % f,
                                              validation_folder)

        if not os.path.isdir(validation_folder_net1):
            raise AssertionError(
                "Validation directory missing: %s. Please rerun validation with `python nnunet_tools/nnunet_fold_val.py --config {config_path} --model_path {model_path} --precision fp16 --save_dir {save_dir} --val_save_folder {val predicted save dir}`"
                % validation_folder_net1)
        if not os.path.isdir(validation_folder_net2):
            raise AssertionError(
                "Validation directory missing: %s. Please rerun validation with `python nnunet_tools/nnunet_fold_val.py --config {config_path} --model_path {model_path} --precision fp16 --save_dir {save_dir} --val_save_folder {val predicted save dir}`"
                % validation_folder_net2)

        if not os.path.isfile(
                os.path.join(validation_folder_net1, 'summary.json')):
            raise AssertionError(
                "Validation directory incomplete: %s. Please rerun validation with `python nnunet_tools/nnunet_fold_val.py --config {config_path} --model_path {model_path} --precision fp16 --save_dir {save_dir} --val_save_folder {val predicted save dir}`"
                % validation_folder_net1)
        if not os.path.isfile(
                os.path.join(validation_folder_net2, 'summary.json')):
            raise AssertionError(
                "Validation directory missing: %s. Please rerun validation with `python nnunet_tools/nnunet_fold_val.py --config {config_path} --model_path {model_path} --precision fp16 --save_dir {save_dir} --val_save_folder {val predicted save dir}`"
                % validation_folder_net2)

        patient_identifiers1_npz = [
            file[:-4] for file in os.listdir(validation_folder_net1)
            if file.endswith('npz')
        ]
        patient_identifiers1_npz.sort()
        patient_identifiers2_npz = [
            file[:-4] for file in os.listdir(validation_folder_net2)
            if file.endswith('npz')
        ]
        patient_identifiers2_npz.sort()

        patient_identifiers1_nii = [
            file for file in os.listdir(validation_folder_net1)
            if file.endswith('nii.gz')
        ]
        patient_identifiers1_nii.sort()
        patient_identifiers1_nii = [
            i[:-7] for i in patient_identifiers1_nii
            if not i.endswith("noPostProcess.nii.gz") and not i.endswith(
                '_postprocessed.nii.gz')
        ]

        patient_identifiers2_nii = [
            file for file in os.listdir(validation_folder_net2)
            if file.endswith('nii.gz')
        ]
        patient_identifiers2_nii.sort()
        patient_identifiers2_nii = [
            i[:-7] for i in patient_identifiers2_nii
            if not i.endswith("noPostProcess.nii.gz") and not i.endswith(
                '_postprocessed.nii.gz')
        ]

        if not all(
            [i in patient_identifiers1_npz for i in patient_identifiers1_nii]):
            raise AssertionError(
                "Missing npz files in folder %s. Please run the validation for all models and folds with the '--save_npz' flag."
                % (validation_folder_net1))
        if not all(
            [i in patient_identifiers2_npz for i in patient_identifiers2_nii]):
            raise AssertionError(
                "Missing npz files in folder %s. Please run the validation for all models and folds with the '--save_npz' flag."
                % (validation_folder_net2))

        assert all([
            i == j
            for i, j in zip(patient_identifiers1_npz, patient_identifiers2_npz)
        ]), "The number of npz file in {} and {} are not matched.".format(
            validation_folder_net1, validation_folder_net2)

        os.makedirs(output_folder, exist_ok=True)

        for p in patient_identifiers1_npz:
            files.append([
                os.path.join(validation_folder_net1, p + '.npz'),
                os.path.join(validation_folder_net2, p + '.npz')
            ])
            property_files.append(
                [os.path.join(validation_folder_net1, p) + ".pkl"])
            out_files.append(os.path.join(output_folder, p + ".nii.gz"))
            gt_segmentations.append(
                os.path.join(folder_with_gt_segs, p + ".nii.gz"))

    p = Pool(threads)
    p.starmap(merge_files,
              zip(files, property_files, out_files, [override] * len(out_files),
                  [False] * len(out_files)))
    p.close()
    p.join()

    if not os.path.isfile(os.path.join(output_folder, "summary.json")) and len(
            out_files) > 0:
        aggregate_scores(
            tuple(zip(out_files, gt_segmentations)),
            labels=plans['all_classes'],
            json_output_file=os.path.join(output_folder, "summary.json"),
            num_threads=threads)
    print("runing postprocessing...")
    determine_postprocessing(
        output_folder_base,
        folder_with_gt_segs,
        "ensembled_raw",
        "temp",
        "ensembled_postprocessed",
        threads,
        dice_threshold=0)
    print("postprocessing done.")


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    if args.gt_dir is None:
        merge(
            args.ensemble_folds,
            args.output_folder,
            threads=args.num_threads,
            postprocessing_file=args.postprocessing_json_path,
            store_npz=args.save_npz)
    else:
        assert args.plan_path is not None, "If ensemble val folders, please set plan_path."
        assert len(args.ensemble_folds
                   ) >= 2, "The number of ensemble folders must greater than 2."

        for folder1, folder2 in combinations(args.ensemble_folds, 2):
            ensemble_name = "ensemble_" + folder1.split('/')[
                -1] + "__" + folder2.split('/')[-1]
            output_folder_base = os.path.join(args.output_folder, "ensembles",
                                              ensemble_name)
            os.makedirs(output_folder_base, exist_ok=True)

            print("ensembling fold: ", folder1, folder2)
            ensemble_val(
                folder1,
                folder2,
                output_folder_base,
                args.plan_path,
                "validation_raw",
                args.folds,
                args.gt_dir,
                threads=args.num_threads,
                override=True)
            print(folder1, folder2, "ensemble over.")


if __name__ == '__main__':
    args = parse_args()
    main(args)

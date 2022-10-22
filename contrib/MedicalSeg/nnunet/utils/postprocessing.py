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
import shutil
import json
import numpy as np
import SimpleITK as sitk
from multiprocessing.pool import Pool
from scipy.ndimage import label
from copy import deepcopy

from .evaluator import aggregate_scores


def copy_geometry(image: sitk.Image, ref: sitk.Image):
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image


def remove_all_but_the_largest_connected_component(
        image: np.ndarray,
        for_which_classes: list,
        volume_per_voxel: float,
        minimum_valid_object_size: dict=None):
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        lmap, num_objects = label(mask.astype(int))
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (
                lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None
        if num_objects > 0:
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                if object_sizes[object_id] != maximum_size:
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[
                            object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c],
                                                     object_sizes[object_id])
    return image, largest_removed, kept_size


def load_remove_save(input_file: str,
                     output_file: str,
                     for_which_classes: list,
                     minimum_valid_object_size: dict=None):
    img_in = sitk.ReadImage(input_file)
    img_npy = sitk.GetArrayFromImage(img_in)
    volume_per_voxel = float(np.prod(img_in.GetSpacing(), dtype=np.float64))
    image, largest_removed, kept_size = remove_all_but_the_largest_connected_component(
        img_npy, for_which_classes, volume_per_voxel, minimum_valid_object_size)
    img_out_itk = sitk.GetImageFromArray(image)
    img_out_itk = copy_geometry(img_out_itk, img_in)
    sitk.WriteImage(img_out_itk, output_file)
    return largest_removed, kept_size


def determine_postprocessing(base,
                             gt_labels_folder,
                             raw_subfolder_name="validation_raw",
                             temp_folder="temp",
                             final_subf_name="validation_final",
                             processes=2,
                             dice_threshold=0,
                             debug=False,
                             advanced_postprocessing=False,
                             pp_filename="postprocessing.json"):

    classes = [
        int(i)
        for i in load_json(
            os.path.join(base, raw_subfolder_name, "summary.json"))['results'][
                'mean'].keys() if int(i) != 0
    ]

    folder_all_classes_as_fg = os.path.join(base, temp_folder + "_allClasses")
    folder_per_class = os.path.join(base, temp_folder + "_perClass")

    if os.path.isdir(folder_all_classes_as_fg):
        shutil.rmtree(folder_all_classes_as_fg)
    if os.path.isdir(folder_per_class):
        shutil.rmtree(folder_per_class)

    p = Pool(processes)
    assert os.path.isfile(
        os.path.join(base, raw_subfolder_name, "summary.json")
    ), "{} does not contain a summary.json.".format(
        os.path.join(base, raw_subfolder_name))

    fnames = []
    for f_name in os.listdir(os.path.join(base, raw_subfolder_name)):
        if os.path.isfile(os.path.join(base, raw_subfolder_name,
                                       f_name)) and f_name.endswith('.nii.gz'):
            fnames.append(f_name)

    os.makedirs(folder_all_classes_as_fg, exist_ok=True)
    os.makedirs(folder_per_class, exist_ok=True)
    os.makedirs(os.path.join(base, final_subf_name), exist_ok=True)

    pp_results = {}
    pp_results['dc_per_class_raw'] = {}
    pp_results['dc_per_class_pp_all'] = {}
    pp_results['dc_per_class_pp_per_class'] = {}
    pp_results['for_which_classes'] = []
    pp_results['min_valid_object_sizes'] = {}

    validation_result_raw = load_json(
        os.path.join(base, raw_subfolder_name, "summary.json"))['results']
    pp_results['num_samples'] = len(validation_result_raw['all'])
    validation_result_raw = validation_result_raw['mean']

    if advanced_postprocessing:
        results = []
        for f in fnames:
            predicted_segmentation = os.path.join(base, raw_subfolder_name, f)
            output_file = os.path.join(folder_all_classes_as_fg, f)
            results.append(
                p.starmap_async(load_remove_save, ((
                    predicted_segmentation, output_file, (classes, )), )))
        results = [i.get() for i in results]

        max_size_removed = {}
        min_size_kept = {}
        for tmp in results:
            mx_rem, min_kept = tmp[0]
            for k in mx_rem:
                if mx_rem[k] is not None:
                    if max_size_removed.get(k) is None:
                        max_size_removed[k] = mx_rem[k]
                    else:
                        max_size_removed[k] = max(max_size_removed[k],
                                                  mx_rem[k])
            for k in min_kept:
                if min_kept[k] is not None:
                    if min_size_kept.get(k) is None:
                        min_size_kept[k] = min_kept[k]
                    else:
                        min_size_kept[k] = min(min_size_kept[k], min_kept[k])

        print("foreground vs background, smallest valid object size was",
              min_size_kept[tuple(classes)])
        print("removing only objects smaller than that...")

    else:
        min_size_kept = None

    pred_gt_tuples = []
    results = []
    for f in fnames:
        predicted_segmentation = os.path.join(base, raw_subfolder_name, f)
        output_file = os.path.join(folder_all_classes_as_fg, f)
        results.append(
            p.starmap_async(load_remove_save, (
                (predicted_segmentation, output_file, (classes, ), min_size_kept
                 ), )))
        pred_gt_tuples.append([output_file, os.path.join(gt_labels_folder, f)])

    _ = [i.get() for i in results]
    _ = aggregate_scores(
        pred_gt_tuples,
        labels=classes,
        json_output_file=os.path.join(folder_all_classes_as_fg, "summary.json"),
        json_author="medicalseg",
        num_threads=processes)
    validation_result_PP_test = load_json(
        os.path.join(folder_all_classes_as_fg, "summary.json"))['results'][
            'mean']

    for c in classes:
        dc_raw = validation_result_raw[str(c)]['Dice']
        dc_pp = validation_result_PP_test[str(c)]['Dice']
        pp_results['dc_per_class_raw'][str(c)] = dc_raw
        pp_results['dc_per_class_pp_all'][str(c)] = dc_pp

    do_fg_cc = False
    comp = [
        pp_results['dc_per_class_pp_all'][str(cl)] >
        (pp_results['dc_per_class_raw'][str(cl)] + dice_threshold)
        for cl in classes
    ]
    before = np.mean(
        [pp_results['dc_per_class_raw'][str(cl)] for cl in classes])
    after = np.mean(
        [pp_results['dc_per_class_pp_all'][str(cl)] for cl in classes])
    print("Foreground vs background")
    print("before:", before)
    print("after: ", after)
    if any(comp):
        any_worse = any([
            pp_results['dc_per_class_pp_all'][str(cl)] <
            pp_results['dc_per_class_raw'][str(cl)] for cl in classes
        ])
        if not any_worse:
            pp_results['for_which_classes'].append(classes)
            if min_size_kept is not None:
                pp_results['min_valid_object_sizes'].update(
                    deepcopy(min_size_kept))
            do_fg_cc = True
            print(
                "Removing all but the largest foreground region improved results!"
            )
            print('for_which_classes', classes)
            print('min_valid_object_sizes', min_size_kept)
    else:
        pass

    if len(classes) > 1:
        if do_fg_cc:
            source = folder_all_classes_as_fg
        else:
            source = os.path.join(base, raw_subfolder_name)

        if advanced_postprocessing:
            results = []
            for f in fnames:
                predicted_segmentation = os.path.join(source, f)
                output_file = os.path.join(folder_per_class, f)
                results.append(
                    p.starmap_async(load_remove_save, ((
                        predicted_segmentation, output_file, classes), )))
            results = [i.get() for i in results]

            max_size_removed = {}
            min_size_kept = {}
            for tmp in results:
                mx_rem, min_kept = tmp[0]
                for k in mx_rem:
                    if mx_rem[k] is not None:
                        if max_size_removed.get(k) is None:
                            max_size_removed[k] = mx_rem[k]
                        else:
                            max_size_removed[k] = max(max_size_removed[k],
                                                      mx_rem[k])
                for k in min_kept:
                    if min_kept[k] is not None:
                        if min_size_kept.get(k) is None:
                            min_size_kept[k] = min_kept[k]
                        else:
                            min_size_kept[k] = min(min_size_kept[k],
                                                   min_kept[k])
            print(
                "classes treated separately, smallest valid object sizes are {}, removing only objects smaller than that.".
                format(min_size_kept))
        else:
            min_size_kept = None

        pred_gt_tuples = []
        results = []
        for f in fnames:
            predicted_segmentation = os.path.join(source, f)
            output_file = os.path.join(folder_per_class, f)
            results.append(
                p.starmap_async(load_remove_save, (
                    (predicted_segmentation, output_file, classes, min_size_kept
                     ), )))
            pred_gt_tuples.append(
                [output_file, os.path.join(gt_labels_folder, f)])

        _ = [i.get() for i in results]
        _ = aggregate_scores(
            pred_gt_tuples,
            labels=classes,
            json_output_file=os.path.join(folder_per_class, "summary.json"),
            json_author="medicalseg",
            num_threads=processes)

        if do_fg_cc:
            old_res = deepcopy(validation_result_PP_test)
        else:
            old_res = validation_result_raw
        validation_result_PP_test = load_json(
            os.path.join(folder_per_class, "summary.json"))['results']['mean']

        for c in classes:
            dc_raw = old_res[str(c)]['Dice']
            dc_pp = validation_result_PP_test[str(c)]['Dice']
            pp_results['dc_per_class_pp_per_class'][str(c)] = dc_pp
            print(c)
            print("before:", dc_raw)
            print("after: ", dc_pp)

            if dc_pp > (dc_raw + dice_threshold):
                pp_results['for_which_classes'].append(int(c))
                if min_size_kept is not None:
                    pp_results['min_valid_object_sizes'].update({
                        c: min_size_kept[c]
                    })
                print(
                    "Removing all but the largest region for class %d improved results!"
                    % c)
                print('min_valid_object_sizes', min_size_kept)
    else:
        print(
            "Only one class present, no need to do each class separately as this is covered in fg vs bg"
        )

    if not advanced_postprocessing:
        pp_results['min_valid_object_sizes'] = None

    print("done")
    print("for which classes:")
    print(pp_results['for_which_classes'])
    print("min_object_sizes")
    print(pp_results['min_valid_object_sizes'])

    pp_results['validation_raw'] = raw_subfolder_name
    pp_results['validation_final'] = final_subf_name

    pred_gt_tuples = []
    results = []
    for f in fnames:
        predicted_segmentation = os.path.join(base, raw_subfolder_name, f)
        output_file = os.path.join(base, final_subf_name, f)
        results.append(
            p.starmap_async(load_remove_save, ((
                predicted_segmentation, output_file, pp_results[
                    'for_which_classes'], pp_results['min_valid_object_sizes']),
                                               )))

        pred_gt_tuples.append([output_file, os.path.join(gt_labels_folder, f)])

    _ = [i.get() for i in results]
    _ = aggregate_scores(
        pred_gt_tuples,
        labels=classes,
        json_output_file=os.path.join(base, final_subf_name, "summary.json"),
        json_author="medicalseg",
        num_threads=processes)
    pp_results['min_valid_object_sizes'] = str(pp_results[
        'min_valid_object_sizes'])
    save_json(pp_results, os.path.join(base, pp_filename))

    if not debug:
        shutil.rmtree(folder_per_class)
        shutil.rmtree(folder_all_classes_as_fg)

    p.close()
    p.join()
    print("post processing done.")


def load_json(file: str):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def save_json(obj, file: str, indent: int=4, sort_keys: bool=True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

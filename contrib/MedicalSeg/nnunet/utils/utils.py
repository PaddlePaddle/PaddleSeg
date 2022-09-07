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
import pickle
import numpy as np
import SimpleITK as sitk
from copy import deepcopy
from typing import Union, Tuple

from tools.preprocess_utils.preprocessing import get_do_separate_z, get_lowres_axis, resample_data_or_seg


def resample_and_save(seg_map,
                      target_shape,
                      output_file,
                      force_separate_z=False,
                      interpolation_order=1,
                      interpolation_order_z=0,
                      remove_seg=True):
    if isinstance(seg_map, str):
        assert os.path.isfile(
            seg_map
        ), "If isinstance(segmentation_softmax, str) then isfile(segmentation_softmax) must be True."
        del_file = deepcopy(seg_map)
        seg_map = np.load(seg_map)
        if remove_seg:
            os.remove(del_file)

    predicted_new_shape = resample_data_or_seg(
        seg_map,
        target_shape,
        False,
        order=interpolation_order,
        do_separate_z=force_separate_z,
        order_z=interpolation_order_z)
    seg_new_shape = predicted_new_shape.argmax(0)
    np.savez_compressed(output_file, data=seg_new_shape.astype(np.uint8))


def save_segmentation_nifti_from_softmax(
        segmentation_softmax: Union[str, np.ndarray],
        out_fname: str,
        properties_dict: dict,
        order: int=1,
        region_class_order: Tuple[Tuple[int]]=None,
        seg_postprogess_fn: callable=None,
        seg_postprocess_args: tuple=None,
        resampled_npz_fname: str=None,
        non_postprocessed_fname: str=None,
        force_separate_z: bool=None,
        interpolation_order_z: int=0,
        verbose: bool=False):
    if verbose:
        print("force_separate_z:", force_separate_z, "interpolation order:",
              order)

    if isinstance(segmentation_softmax, str):
        assert os.path.isfile(segmentation_softmax), "If isinstance(segmentation_softmax, str) then " \
                                             "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(segmentation_softmax)
        if segmentation_softmax.endswith('.npy'):
            segmentation_softmax = np.load(segmentation_softmax)
        elif segmentation_softmax.endswith('.npz'):
            segmentation_softmax = np.load(segmentation_softmax)['softmax']
        os.remove(del_file)

    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = properties_dict.get('size_after_cropping')
    shape_original_before_cropping = properties_dict.get(
        'original_size_of_raw_data')

    if np.any([
            i != j
            for i, j in zip(
                np.array(current_shape[1:]),
                np.array(shape_original_after_cropping))
    ]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(
                    properties_dict.get('original_spacing'))
            elif get_do_separate_z(
                    properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(
                    properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(
                    properties_dict.get('original_spacing'))
            else:
                lowres_axis = None

        if lowres_axis is not None and len(lowres_axis) != 1:
            do_separate_z = False

        if verbose:
            print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        seg_old_spacing = resample_data_or_seg(
            segmentation_softmax,
            shape_original_after_cropping,
            is_seg=False,
            axis=lowres_axis,
            order=order,
            do_separate_z=do_separate_z,
            order_z=interpolation_order_z)
    else:
        if verbose:
            print("no resampling necessary")
        seg_old_spacing = segmentation_softmax
    if resampled_npz_fname is not None:
        np.savez_compressed(
            resampled_npz_fname, softmax=seg_old_spacing.astype(np.float16))
        # this is needed for ensembling if the nonlinearity is sigmoid
        if region_class_order is not None:
            properties_dict['regions_class_order'] = region_class_order
        with open(resampled_npz_fname[:-4] + ".pkl", 'wb') as f:
            pickle.dump(properties_dict, f)

    if region_class_order is None:
        seg_old_spacing = seg_old_spacing.argmax(0)
    else:
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
        seg_old_spacing = seg_old_spacing_final

    bbox = properties_dict.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping, dtype=np.uint8)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c],
                                 shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:
                     bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    if seg_postprogess_fn is not None:
        seg_old_size_postprocessed = seg_postprogess_fn(
            np.copy(seg_old_size), *seg_postprocess_args)
    else:
        seg_old_size_postprocessed = seg_old_size

    seg_resized_itk = sitk.GetImageFromArray(
        seg_old_size_postprocessed.astype(np.uint8))
    seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)

    if (non_postprocessed_fname is not None) and (
            seg_postprogess_fn is not None):
        seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
        seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
        seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
        seg_resized_itk.SetDirection(properties_dict['itk_direction'])
        sitk.WriteImage(seg_resized_itk, non_postprocessed_fname)


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def pad_nd_image(image,
                 new_shape=None,
                 mode="constant",
                 kwargs=None,
                 return_slicer=False,
                 shape_must_be_divisible_by=None):
    """
    image: Input image.
    new_shape: New shape of image, if None, the image shape will be determined by shape_must_be_divisible_by. Default: None.
    mode: The padding mode for image. Reference to np.pad for more information.
    kwargs: The padding kwargs for image. Reference to np.pad for more information.
    return_slicer: if True then this function will also return what coords you will need to use when cropping back to original shape. Default: False.
    shape_must_be_divisible_by: After applying new_shape, make sure the new shape is divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None). Default: None.
    """
    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None, "shape_must_be_divisible_by must not be None when shape_must_be_divisible_by is None."
        assert isinstance(
            shape_must_be_divisible_by, (list, tuple, np.ndarray)
        ), "shape_must_be_divisible_by can be 'list', 'tuple', 'np.ndarray', but got {}.".format(
            shape_must_be_divisible_by)
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)
    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by,
                          (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(
                new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([
            new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] %
            shape_must_be_divisible_by[i] for i in range(len(new_shape))
        ])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]] * num_axes_nopad + list(
        [list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and
            (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res

    pad_list = np.array(pad_list)
    pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
    slicer = list(slice(*i) for i in pad_list)
    return res, slicer


class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

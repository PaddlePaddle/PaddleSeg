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

import random
import numpy as np

import paddle
from paddle.nn.functional import avg_pool2d, avg_pool3d

from typing import List, Tuple, Union, Callable
from copy import deepcopy
from scipy.ndimage import map_coordinates, fourier_gaussian
from scipy.ndimage.filters import gaussian_filter, gaussian_gradient_magnitude
from scipy.ndimage.morphology import grey_dilation
from skimage.transform import resize
from scipy.ndimage.measurements import label as lb


def resize_segmentation(segmentation, new_shape, order=3):
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(
        new_shape), "New shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(
            segmentation.astype(float),
            new_shape,
            order,
            mode="edge",
            clip=True,
            anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(
                mask.astype(float),
                new_shape,
                order,
                mode="edge",
                clip=True,
                anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


def downsample_seg_for_ds_transform2(seg,
                                     ds_scales=((1, 1, 1), (0.5, 0.5, 0.5),
                                                (0.25, 0.25, 0.25)),
                                     order=0,
                                     axes=None):
    if axes is None:
        axes = list(range(2, len(seg.shape)))
    output = []
    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(seg)
        else:
            new_shape = np.array(seg.shape).astype(float)
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            new_shape = np.round(new_shape).astype(int)
            out_seg = np.zeros(new_shape, dtype=seg.dtype)
            for b in range(seg.shape[0]):
                for c in range(seg.shape[1]):
                    out_seg[b, c] = resize_segmentation(seg[b, c],
                                                        new_shape[2:], order)
            output.append(out_seg)
    return output


def convert_seg_image_to_one_hot_encoding_batched(image, classes=None):
    if classes is None:
        classes = np.unique(image)
    output_shape = [image.shape[0]] + [len(classes)] + list(image.shape[1:])
    out_image = np.zeros(output_shape, dtype=image.dtype)
    for b in range(image.shape[0]):
        for i, c in enumerate(classes):
            out_image[b, i][image[b] == c] = 1
    return out_image


def downsample_seg_for_ds_transform3(seg,
                                     ds_scales=((1, 1, 1), (0.5, 0.5, 0.5),
                                                (0.25, 0.25, 0.25)),
                                     classes=None):
    output = []
    one_hot = paddle.to_tensor(
        convert_seg_image_to_one_hot_encoding_batched(seg, classes))

    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(paddle.to_tensor(seg))
        else:
            kernel_size = tuple(int(1 / i) for i in s)
            stride = kernel_size
            pad = tuple((i - 1) // 2 for i in kernel_size)

            if len(s) == 2:
                pool_op = avg_pool2d
            elif len(s) == 3:
                pool_op = avg_pool3d
            else:
                raise RuntimeError()

            pooled = pool_op(
                one_hot,
                kernel_size,
                stride,
                pad,
                count_include_pad=False,
                ceil_mode=False)
            output.append(pooled)
    return output


def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2)):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]")
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data, sample_seg


def augment_gamma(data_sample,
                  gamma_range=(0.5, 2),
                  invert_image=False,
                  epsilon=1e-7,
                  per_channel=False,
                  retain_stats: Union[bool, Callable[[], bool]]=False):
    if invert_image:
        data_sample = -data_sample

    if not per_channel:
        retain_stats_here = retain_stats() if callable(
            retain_stats) else retain_stats
        if retain_stats_here:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power((
            (data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats_here:
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn
    else:
        for c in range(data_sample.shape[0]):
            retain_stats_here = retain_stats() if callable(
                retain_stats) else retain_stats
            if retain_stats_here:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(
                    max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power((
                (data_sample[c] - minm) / float(rnge + epsilon)),
                                      gamma) * float(rnge + epsilon) + minm
            if retain_stats_here:
                data_sample[c] = data_sample[c] - data_sample[c].mean()
                data_sample[c] = data_sample[c] / (
                    data_sample[c].std() + 1e-8) * sd
                data_sample[c] = data_sample[c] + mn
    if invert_image:
        data_sample = -data_sample
    return data_sample


def augment_linear_downsampling_scipy(data_sample,
                                      zoom_range=(0.5, 1),
                                      per_channel=True,
                                      p_per_channel=1,
                                      channels=None,
                                      order_downsample=1,
                                      order_upsample=0,
                                      ignore_axes=None):
    if not isinstance(zoom_range, (list, tuple, np.ndarray)):
        zoom_range = [zoom_range]

    shp = np.array(data_sample.shape[1:])
    dim = len(shp)
    if not per_channel:
        if isinstance(zoom_range[0], (tuple, list, np.ndarray)):
            assert len(zoom_range) == dim
            zoom = np.array([uniform(i[0], i[1]) for i in zoom_range])
        else:
            zoom = uniform(zoom_range[0], zoom_range[1])

        target_shape = np.round(shp * zoom).astype(int)

        if ignore_axes is not None:
            for i in ignore_axes:
                target_shape[i] = shp[i]

    if channels is None:
        channels = list(range(data_sample.shape[0]))

    for c in channels:
        if np.random.uniform() < p_per_channel:
            if per_channel:
                if isinstance(zoom_range[0], (tuple, list, np.ndarray)):
                    assert len(zoom_range) == dim
                    zoom = np.array([uniform(i[0], i[1]) for i in zoom_range])
                else:
                    zoom = uniform(zoom_range[0], zoom_range[1])

                target_shape = np.round(shp * zoom).astype(int)
                if ignore_axes is not None:
                    for i in ignore_axes:
                        target_shape[i] = shp[i]

            downsampled = resize(
                data_sample[c].astype(float),
                target_shape,
                order=order_downsample,
                mode='edge',
                anti_aliasing=False)
            data_sample[c] = resize(
                downsampled,
                shp,
                order=order_upsample,
                mode='edge',
                anti_aliasing=False)
    return data_sample


def uniform(low, high, size=None):
    if low == high:
        if size is None:
            return low
        else:
            return np.ones(size) * low
    else:
        return np.random.uniform(low, high, size)


def augment_contrast(
        data_sample: np.ndarray,
        contrast_range: Union[Tuple[float, float], Callable[[], float]]=(0.75,
                                                                         1.25),
        preserve_range: bool=True,
        per_channel: bool=True,
        p_per_channel: float=1) -> np.ndarray:
    if not per_channel:
        if callable(contrast_range):
            factor = contrast_range()
        else:
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(
                    max(contrast_range[0], 1), contrast_range[1])

        for c in range(data_sample.shape[0]):
            if np.random.uniform() < p_per_channel:
                mn = data_sample[c].mean()
                if preserve_range:
                    minm = data_sample[c].min()
                    maxm = data_sample[c].max()

                data_sample[c] = (data_sample[c] - mn) * factor + mn

                if preserve_range:
                    data_sample[c][data_sample[c] < minm] = minm
                    data_sample[c][data_sample[c] > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            if np.random.uniform() < p_per_channel:
                if callable(contrast_range):
                    factor = contrast_range()
                else:
                    if np.random.random() < 0.5 and contrast_range[0] < 1:
                        factor = np.random.uniform(contrast_range[0], 1)
                    else:
                        factor = np.random.uniform(
                            max(contrast_range[0], 1), contrast_range[1])

                mn = data_sample[c].mean()
                if preserve_range:
                    minm = data_sample[c].min()
                    maxm = data_sample[c].max()

                data_sample[c] = (data_sample[c] - mn) * factor + mn

                if preserve_range:
                    data_sample[c][data_sample[c] < minm] = minm
                    data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample


def augment_brightness_additive(data_sample,
                                mu: float,
                                sigma: float,
                                per_channel: bool=True,
                                p_per_channel: float=1.):
    if not per_channel:
        rnd_nb = np.random.normal(mu, sigma)
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                data_sample[c] += rnd_nb
    else:
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                rnd_nb = np.random.normal(mu, sigma)
                data_sample[c] += rnd_nb
    return data_sample


def augment_brightness_multiplicative(data_sample,
                                      multiplier_range=(0.5, 2),
                                      per_channel=True):
    multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
    if not per_channel:
        data_sample *= multiplier
    else:
        for c in range(data_sample.shape[0]):
            multiplier = np.random.uniform(multiplier_range[0],
                                           multiplier_range[1])
            data_sample[c] *= multiplier
    return data_sample


def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError(
                "value must be either a single value or a list/tuple of len 2")
        return n_val
    else:
        return value


def augment_gaussian_blur(data_sample: np.ndarray,
                          sigma_range: Tuple[float, float],
                          per_channel: bool=True,
                          p_per_channel: float=1,
                          different_sigma_per_axis: bool=False,
                          p_isotropic: float=0) -> np.ndarray:
    if not per_channel:
        sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                               ((np.random.uniform() < p_isotropic) and
                                                different_sigma_per_axis)) \
            else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
    else:
        sigma = None
    for c in range(data_sample.shape[0]):
        if np.random.uniform() <= p_per_channel:
            if per_channel:
                sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                                       ((np.random.uniform() < p_isotropic) and
                                                        different_sigma_per_axis)) \
                    else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
            data_sample[c] = gaussian_filter(data_sample[c], sigma, order=0)
    return data_sample


def augment_gaussian_noise(data_sample: np.ndarray,
                           noise_variance: Tuple[float, float]=(0, 0.1),
                           p_per_channel: float=1,
                           per_channel: bool=False) -> np.ndarray:
    if not per_channel:
        variance = noise_variance[0] if noise_variance[0] == noise_variance[1] else \
            random.uniform(noise_variance[0], noise_variance[1])
    else:
        variance = None
    for c in range(data_sample.shape[0]):
        if np.random.uniform() < p_per_channel:
            variance_here = variance if variance is not None else \
                noise_variance[0] if noise_variance[0] == noise_variance[1] else \
                    random.uniform(noise_variance[0], noise_variance[1])
            data_sample[c] = data_sample[c] + np.random.normal(
                0.0, variance_here, size=data_sample[c].shape)
    return data_sample


def convert_2d_to_3d_generator(data_dict):
    shp = data_dict['orig_shape_data']
    current_shape = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape(
        (shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    shp = data_dict['orig_shape_seg']
    current_shape_seg = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape(
        (shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    return data_dict


def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords


def elastic_deform_coordinates(coordinates, alpha, sigma):
    n_dim = len(coordinates)
    offsets = []
    for _ in range(n_dim):
        offsets.append(
            gaussian_filter(
                (np.random.random(coordinates.shape[1:]) * 2 - 1),
                sigma,
                mode="constant",
                cval=0) * alpha)
    offsets = np.array(offsets)
    indices = offsets + coordinates
    return indices


def create_matrix_rotation_x_3d(angle, matrix=None):
    rotation_x = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation_x

    return np.dot(matrix, rotation_x)


def create_matrix_rotation_y_3d(angle, matrix=None):
    rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])
    if matrix is None:
        return rotation_y

    return np.dot(matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, matrix=None):
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    if matrix is None:
        return rotation_z

    return np.dot(matrix, rotation_z)


def create_matrix_rotation_2d(angle, matrix=None):
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation

    return np.dot(matrix, rotation)


def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(),
                    rot_matrix).transpose().reshape(coords.shape)
    return coords


def rotate_coords_2d(coords, angle):
    rot_matrix = create_matrix_rotation_2d(angle)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(),
                    rot_matrix).transpose().reshape(coords.shape)
    return coords


def scale_coords(coords, scale):
    if isinstance(scale, (tuple, list, np.ndarray)):
        assert len(scale) == len(coords)
        for i in range(len(scale)):
            coords[i] *= scale[i]
    else:
        coords *= scale
    return coords


def interpolate_img(img,
                    coords,
                    order=3,
                    mode='nearest',
                    cval=0.0,
                    is_seg=False):
    if is_seg and order != 0:
        unique_labels = np.unique(img)
        result = np.zeros(coords.shape[1:], img.dtype)
        for i, c in enumerate(unique_labels):
            res_new = map_coordinates(
                (img == c).astype(float),
                coords,
                order=order,
                mode=mode,
                cval=cval)
            result[res_new >= 0.5] = c
        return result
    else:
        return map_coordinates(
            img.astype(float), coords, order=order, mode=mode,
            cval=cval).astype(img.dtype)


def get_lbs_for_center_crop(crop_size, data_shape):
    lbs = []
    for i in range(len(data_shape) - 2):
        lbs.append((data_shape[i + 2] - crop_size[i]) // 2)
    return lbs


def get_lbs_for_random_crop(crop_size, data_shape, margins):
    lbs = []
    for i in range(len(data_shape) - 2):
        if data_shape[i + 2] - crop_size[i] - margins[i] > margins[i]:
            lbs.append(
                np.random.randint(margins[i], data_shape[i + 2] - crop_size[i] -
                                  margins[i]))
        else:
            lbs.append((data_shape[i + 2] - crop_size[i]) // 2)
    return lbs


def crop(data,
         seg=None,
         crop_size=128,
         margins=(0, 0, 0),
         crop_type="center",
         pad_mode='constant',
         pad_kwargs={'constant_values': 0},
         pad_mode_seg='constant',
         pad_kwargs_seg={'constant_values': 0}):
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError("data has to be either a numpy array or a list")

    data_shape = tuple([len(data)] + list(data[0].shape))
    data_dtype = data[0].dtype
    dim = len(data_shape) - 2

    if seg is not None:
        seg_shape = tuple([len(seg)] + list(seg[0].shape))
        seg_dtype = seg[0].dtype

        if not isinstance(seg, (list, tuple, np.ndarray)):
            raise TypeError("data has to be either a numpy array or a list")

        assert all([i == j for i, j in zip(seg_shape[2:], data_shape[2:])]), "data and seg must have the same spatial " \
                                                                             "dimensions. Data: {}, seg: {}.".format(data_shape, seg_shape)

    if type(crop_size) not in (tuple, list, np.ndarray):
        crop_size = [crop_size] * dim
    else:
        assert len(crop_size) == len(
            data_shape) - 2, "If you provide a list/tuple as center crop make sure it has the same dimension as your " \
                             "data (2d/3d)"

    if not isinstance(margins, (np.ndarray, tuple, list)):
        margins = [margins] * dim

    data_return = np.zeros(
        [data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype)
    if seg is not None:
        seg_return = np.zeros(
            [seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg_dtype)
    else:
        seg_return = None

    for b in range(data_shape[0]):
        data_shape_here = [data_shape[0]] + list(data[b].shape)
        if seg is not None:
            seg_shape_here = [seg_shape[0]] + list(seg[b].shape)

        if crop_type == "center":
            lbs = get_lbs_for_center_crop(crop_size, data_shape_here)
        elif crop_type == "random":
            lbs = get_lbs_for_random_crop(crop_size, data_shape_here, margins)
        else:
            raise NotImplementedError(
                "crop_type must be either center or random")

        need_to_pad = [[0, 0]] + [[
            abs(min(0, lbs[d])),
            abs(min(0, data_shape_here[d + 2] - (lbs[d] + crop_size[d])))
        ] for d in range(dim)]

        ubs = [
            min(lbs[d] + crop_size[d], data_shape_here[d + 2])
            for d in range(dim)
        ]
        lbs = [max(0, lbs[d]) for d in range(dim)]

        slicer_data = [slice(0, data_shape_here[1])
                       ] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        data_cropped = data[b][tuple(slicer_data)]

        if seg_return is not None:
            slicer_seg = [slice(0, seg_shape_here[1])
                          ] + [slice(lbs[d], ubs[d]) for d in range(dim)]
            seg_cropped = seg[b][tuple(slicer_seg)]

        if any([i > 0 for j in need_to_pad for i in j]):
            data_return[b] = np.pad(data_cropped, need_to_pad, pad_mode,
                                    **pad_kwargs)
            if seg_return is not None:
                seg_return[b] = np.pad(seg_cropped, need_to_pad, pad_mode_seg,
                                       **pad_kwargs_seg)
        else:
            data_return[b] = data_cropped
            if seg_return is not None:
                seg_return[b] = seg_cropped

    return data_return, seg_return


def random_crop(data, seg=None, crop_size=128, margins=[0, 0, 0]):
    return crop(data, seg, crop_size, margins, 'random')


def center_crop(data, crop_size, seg=None):
    return crop(data, seg, crop_size, 0, 'center')


def augment_spatial(data,
                    seg,
                    patch_size,
                    patch_center_dist_from_border=30,
                    do_elastic_deform=True,
                    alpha=(0., 1000.),
                    sigma=(10., 13.),
                    do_rotation=True,
                    angle_x=(0, 2 * np.pi),
                    angle_y=(0, 2 * np.pi),
                    angle_z=(0, 2 * np.pi),
                    do_scale=True,
                    scale=(0.75, 1.25),
                    border_mode_data='nearest',
                    border_cval_data=0,
                    order_data=3,
                    border_mode_seg='constant',
                    border_cval_seg=0,
                    order_seg=0,
                    random_crop=True,
                    p_el_per_sample=1,
                    p_scale_per_sample=1,
                    p_rot_per_sample=1,
                    independent_scale_for_each_axis=False,
                    p_rot_per_axis: float=1,
                    p_independent_scale_per_axis: int=1):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros(
                (seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]),
                dtype=np.float32)
        else:
            seg_result = np.zeros(
                (seg.shape[0], seg.shape[1], patch_size[0], patch_size[1],
                 patch_size[2]),
                dtype=np.float32)

    if dim == 2:
        data_result = np.zeros(
            (data.shape[0], data.shape[1], patch_size[0], patch_size[1]),
            dtype=np.float32)
    else:
        data_result = np.zeros(
            (data.shape[0], data.shape[1], patch_size[0], patch_size[1],
             patch_size[2]),
            dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        if do_elastic_deform and np.random.uniform() < p_el_per_sample:
            a = np.random.uniform(alpha[0], alpha[1])
            s = np.random.uniform(sigma[0], sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)
            modified_coords = True

        if do_rotation and np.random.uniform() < p_rot_per_sample:

            if np.random.uniform() <= p_rot_per_axis:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            else:
                a_x = 0

            if dim == 3:
                if np.random.uniform() <= p_rot_per_axis:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                else:
                    a_y = 0

                if np.random.uniform() <= p_rot_per_axis:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                else:
                    a_z = 0

                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)
            modified_coords = True

        if do_scale and np.random.uniform() < p_scale_per_sample:
            if independent_scale_for_each_axis and np.random.uniform(
            ) < p_independent_scale_per_axis:
                sc = []
                for _ in range(dim):
                    if np.random.random() < 0.5 and scale[0] < 1:
                        sc.append(np.random.uniform(scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
            else:
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc = np.random.uniform(scale[0], 1)
                else:
                    sc = np.random.uniform(max(scale[0], 1), scale[1])

            coords = scale_coords(coords, sc)
            modified_coords = True

        if modified_coords:
            for d in range(dim):
                if random_crop:
                    ctr = np.random.uniform(
                        patch_center_dist_from_border[d],
                        data.shape[d + 2] - patch_center_dist_from_border[d])
                else:
                    ctr = data.shape[d + 2] / 2. - 0.5
                coords[d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(
                    data[sample_id, channel_id],
                    coords,
                    order_data,
                    border_mode_data,
                    cval=border_cval_data)
            if seg is not None:
                for channel_id in range(seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(
                        seg[sample_id, channel_id],
                        coords,
                        order_seg,
                        border_mode_seg,
                        cval=border_cval_seg,
                        is_seg=True)
        else:
            if seg is None:
                s = None
            else:
                s = seg[sample_id:sample_id + 1]
            if random_crop:
                margin = [
                    patch_center_dist_from_border[d] - patch_size[d] // 2
                    for d in range(dim)
                ]
                d, s = random_crop(data[sample_id:sample_id + 1], s, patch_size,
                                   margin)
            else:
                d, s = center_crop(data[sample_id:sample_id + 1], patch_size, s)
            data_result[sample_id] = d[0]
            if seg is not None:
                seg_result[sample_id] = s[0]
    return data_result, seg_result


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape(
        (shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp
    shp = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape(
        (shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_seg'] = shp
    return data_dict

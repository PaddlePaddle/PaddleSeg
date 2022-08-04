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
from typing import List, Tuple, Union, Callable
import collections
import numbers
import random

import numpy as np
import scipy
import scipy.ndimage
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter, gaussian_gradient_magnitude
from skimage.transform import resize

def resize_3d(img, size, order=1):
    r"""Resize the input numpy ndarray to the given size.
    Args:
        img (numpy ndarray): Image to be resized.
        size
        order (int, optional): Desired order of scipy.zoom . Default is 1
    Returns:
        Numpy Array
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or
            (isinstance(size, collections.abc.Iterable) and len(size) == 3)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))
    d, h, w = img.shape[0], img.shape[1], img.shape[2]

    if isinstance(size, int):
        if min(d, h, w) == size:
            return img
        ow = int(size * w / min(d, h, w))
        oh = int(size * h / min(d, h, w))
        od = int(size * d / min(d, h, w))
    else:
        ow, oh, od = size[2], size[1], size[0]

    if img.ndim == 3:
        resize_factor = np.array([od, oh, ow]) / img.shape
        output = scipy.ndimage.zoom(
            img, resize_factor, mode='nearest', order=order)
    elif img.ndim == 4:
        resize_factor = np.array([od, oh, ow, img.shape[3]]) / img.shape
        output = scipy.ndimage.zoom(
            img, resize_factor, mode='nearest', order=order)
    return output


def crop_3d(img, i, j, k, d, h, w):
    """Crop the given PIL Image.
    Args:
        img (numpy ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        k:
        d:
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        numpy ndarray: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    return img[i:i + d, j:j + h, k:k + w]


def flip_3d(img, axis):
    """
    axis: int
          0 - flip along Depth  (z-axis)
          1 - flip along Height (y-axis)
          2 - flip along Width  (x-axis)
    """
    img = np.flip(img, axis)
    return img


def rotate_3d(img, r_plane, angle, order=1, cval=0):
    """
    rotate 3D image by r_plane and angle.

    r_plane (2-list): rotate planes by axis, i.e, [0, 1] or [1, 2] or [0, 2]
    angle (int): rotate degrees
    """
    img = scipy.ndimage.rotate(
        img, angle=angle, axes=r_plane, order=order, cval=cval, reshape=False)
    return img


def resized_crop_3d(img, i, j, k, d, h, w, size, interpolation):
    """
    适用于3D数据的resize + crop
    """
    assert _is_numpy_image(img), 'img should be numpy image'
    img = crop_3d(img, i, j, k, d, h, w)
    img = resize_3d(img, size, order=interpolation)
    return img


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3, 4})


def extract_connect_compoent(binary_mask, minimum_volume=0):
    """
    extract connect compoent from binary mask
    binary mask -> mask w/ [0, 1, 2, ...]
    0 - background
    1 - foreground instance #1 (start with 1)
    2 - foreground instance #2
    """
    assert len(np.unique(binary_mask)) < 3, \
        "Only binary mask is accepted, got mask with {}.".format(np.unique(binary_mask).tolist())
    instance_mask = sitk.GetArrayFromImage(
        sitk.RelabelComponent(
            sitk.ConnectedComponent(sitk.GetImageFromArray(binary_mask)),
            minimumObjectSize=minimum_volume))
    return instance_mask


def rotate_4d(img, r_plane, angle, order=1, cval=0):
    """
    rotate 4D image by r_plane and angle.
    r_plane (2-list): rotate planes by axis, i.e, [0, 1] or [1, 2] or [0, 2]
    angle (int): rotate degrees
    """
    img = scipy.ndimage.rotate(
        img,
        angle=angle,
        axes=tuple(r_plane),
        order=order,
        cval=cval,
        reshape=False)
    return img


def crop_4d(img, i, j, k, d, h, w):
    """Crop the given PIL Image.
    Args:
        img (numpy ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        k:
        d:
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        numpy ndarray: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    return img[:, i:i + d, j:j + h, k:k + w]

def augment_gaussian_noise(data_sample: np.ndarray, noise_variance: Tuple[float, float] = (0, 0.1),
                           p_per_channel: float = 1, per_channel: bool = False) -> np.ndarray:
    if not per_channel:
        variance = noise_variance[0] if noise_variance[0] == noise_variance[1] else \
            random.uniform(noise_variance[0], noise_variance[1])
    else:
        variance = None
    for c in range(data_sample.shape[0]):
        if np.random.uniform() < p_per_channel:
            # lol good luck reading this
            variance_here = variance if variance is not None else \
                noise_variance[0] if noise_variance[0] == noise_variance[1] else \
                    random.uniform(noise_variance[0], noise_variance[1])
            # bug fixed: https://github.com/MIC-DKFZ/batchgenerators/issues/86
            data_sample[c] = data_sample[c] + np.random.normal(0.0, variance_here, size=data_sample[c].shape)
    return data_sample

def uniform(low, high, size=None):
    if low == high:
        if size is None:
            return low
        else:
            return np.ones(size) * low
    else:
        return np.random.uniform(low, high, size)
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
            raise RuntimeError("value must be either a single value or a list/tuple of len 2")
        return n_val
    else:
        return value

def augment_gaussian_blur(data_sample: np.ndarray, sigma_range: Tuple[float, float], per_channel: bool = True,
                          p_per_channel: float = 1, different_sigma_per_axis: bool = False,
                          p_isotropic: float = 0) -> np.ndarray:
    if not per_channel:
        # Godzilla Had a Stroke Trying to Read This and F***ing Died
        # https://i.kym-cdn.com/entries/icons/original/000/034/623/Untitled-3.png
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

def augment_brightness_multiplicative(data_sample, multiplier_range=(0.5, 2), per_channel=True):
    multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
    if not per_channel:
        data_sample *= multiplier
    else:
        for c in range(data_sample.shape[0]):
            multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
            data_sample[c] *= multiplier
    return data_sample

def augment_contrast(data_sample: np.ndarray,
                     contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
                     preserve_range: bool = True,
                     per_channel: bool = True,
                     p_per_channel: float = 1) -> np.ndarray:
    if not per_channel:
        if callable(contrast_range):
            factor = contrast_range()
        else:
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])

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
                        factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])

                mn = data_sample[c].mean()
                if preserve_range:
                    minm = data_sample[c].min()
                    maxm = data_sample[c].max()

                data_sample[c] = (data_sample[c] - mn) * factor + mn

                if preserve_range:
                    data_sample[c][data_sample[c] < minm] = minm
                    data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample

def augment_linear_downsampling_scipy(data_sample, zoom_range=(0.5, 1), per_channel=True, p_per_channel=1,
                                      channels=None, order_downsample=1, order_upsample=0, ignore_axes=None):
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

            downsampled = resize(data_sample[c].astype(float), target_shape, order=order_downsample, mode='edge',
                                 anti_aliasing=False)
            data_sample[c] = resize(downsampled, shp, order=order_upsample, mode='edge',
                                    anti_aliasing=False)

    return data_sample

def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats: Union[bool, Callable[[], bool]] = False):
    if invert_image:
        data_sample = - data_sample

    if not per_channel:
        retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
        if retain_stats_here:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats_here:
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn
    else:
        for c in range(data_sample.shape[0]):
            retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
            if retain_stats_here:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats_here:
                data_sample[c] = data_sample[c] - data_sample[c].mean()
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
                data_sample[c] = data_sample[c] + mn
    if invert_image:
        data_sample = - data_sample
    return data_sample

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
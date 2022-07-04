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

import collections
import numbers
import random

import numpy as np
import scipy
import scipy.ndimage
import SimpleITK as sitk


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

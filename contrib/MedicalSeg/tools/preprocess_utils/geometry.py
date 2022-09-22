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
import sys
import os
from skimage.transform import resize

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
import tools.preprocess_utils.global_var as global_var

gpu_tag = global_var.get_value('USE_GPU')
if gpu_tag:
    import cupy as np
    import cupyx.scipy as scipy
    import cupyx.scipy.ndimage
else:
    import numpy as np
    import scipy


def resample(image,
             spacing=None,
             new_spacing=[1.0, 1.0, 1.0],
             new_shape=None,
             order=1):
    """
    Resample image from the original spacing to new_spacing, e.g. 1x1x1

    image(numpy array): 3D numpy array of raw HU values from CT series in [z, y, x] order.
    spacing(list|tuple): float * 3, raw CT spacing in [z, y, x] order.
    new_spacing: float * 3, new spacing used for resample, typically 1x1x1,
        which means standardizing the raw CT with different spacing all into
        1x1x1 mm.
    new_shape(list|tuple): the new shape of resampled numpy array.
    order(int): order for resample function scipy.ndimage.zoom

    return: 3D binary numpy array with the same shape of the image after,
        resampling. The actual resampling spacing is also returned.
    """

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if new_shape is None:
        spacing = np.array([spacing[0], spacing[1], spacing[2]])
        new_shape = np.round(image.shape * spacing / new_spacing)
    else:
        new_shape = np.array(new_shape)
        if spacing is not None and len(spacing) == 4:
            spacing = spacing[1:]
        new_spacing = tuple((image.shape / new_shape) *
                            spacing) if spacing is not None else None

    resize_factor = new_shape / np.array(image.shape)

    image_new = scipy.ndimage.zoom(
        image, resize_factor, mode='nearest', order=order)

    return image_new, new_spacing


def resize_segmentation(segmentation, new_shape, order=3):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to
    one_hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(
        new_shape), "new shape must have same dimensionality as segmentation"
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


def resize_image(image, new_shape, order=3, cval=0):
    kwargs = {'mode': 'edge', 'anti_aliasing': False}
    return resize(image, new_shape, order, cval=cval, **kwargs)

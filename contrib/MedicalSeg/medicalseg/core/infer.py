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

import collections.abc
import paddle as torch
import paddle
import paddle.nn.functional as F
import math


def get_reverse_list(ori_shape, transforms):
    """
    get reverse list of transform.
    Args:
        ori_shape (list): Origin shape of image.
        transforms (list): List of transform.
    Returns:
        list: List of tuple, there are two format:
            ('resize', (h, w)) The image shape before resize,
            ('padding', (h, w)) The image shape before padding.
    """
    reverse_list = []
    d, h, w = ori_shape[0], ori_shape[1], ori_shape[2]
    for op in transforms:
        if op.__class__.__name__ in ['Resize3D']:
            reverse_list.append(('resize', (d, h, w)))
            d, h, w = op.size[0], op.size[1], op.size[2]

    return reverse_list


def reverse_transform(pred, ori_shape, transforms, mode='trilinear'):
    """recover pred to origin shape"""
    reverse_list = get_reverse_list(ori_shape, transforms)
    intTypeList = [paddle.int8, paddle.int16, paddle.int32, paddle.int64]
    dtype = pred.dtype
    for item in reverse_list[::-1]:
        if item[0] == 'resize':
            d, h, w = item[1][0], item[1][1], item[1][2]
            if paddle.get_device() == 'cpu' and dtype in intTypeList:
                pred = paddle.cast(pred, 'float32')
                pred = F.interpolate(pred, (d, h, w), mode=mode)
                pred = paddle.cast(pred, dtype)
            else:
                pred = F.interpolate(pred, (d, h, w), mode=mode)
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return pred


def inference(model, im, ori_shape=None, transforms=None, sw_num=None):
    """
    Inference for image.
    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        ori_shape (list): Origin shape of image.
        transforms (list): Transforms for image.
        sw_num(int):sw_num
    Returns:
        Tensor: If ori_shape is not None, a prediction with shape (1, 1, d, h, w) is returned.
            If ori_shape is None, a logit with shape (1, num_classes, d, h, w) is returned.
    """
    if hasattr(model, 'data_format') and model.data_format == 'NDHWC':
        im = im.transpose((0, 2, 3, 4, 1))

    # If you want to use sliding window inference, make sure the model has the img_shape parameter

    if sw_num:
        logits = sliding_window_inference(im, model.img_shape, sw_num, model)
    else:
        logits = model(im)
    if not isinstance(logits, collections.abc.Sequence):
        raise TypeError(
            "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
            .format(type(logits)))
    logit = logits[0]

    if hasattr(model, 'data_format') and model.data_format == 'NDHWC':
        logit = logit.transpose((0, 4, 1, 2, 3))

    if ori_shape is not None and ori_shape != logit.shape[2:]:
        logit = reverse_transform(logit, ori_shape, transforms, mode='bilinear')

    pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')

    return pred, logit


# Implementation of this function is borrowed and modified
# (from torch to paddle) from here:
# https://docs.monai.io/en/0.1.0/_modules/monai/data/sliding_window_inference.html#sliding_window_inference


def dense_patch_slices(image_size, patch_size, scan_interval):
    """
    Enumerate all slices defining 2D/3D patches of size `patch_size` from an `image_size` input image.
    Args:
        image_size (tuple of int): dimensions of image to iterate over
        patch_size (tuple of int): size of patches to generate slices
        scan_interval (tuple of int): dense patch sampling interval
    Returns:
        a list of slice objects defining each patch
    """
    num_spatial_dims = len(image_size)
    if num_spatial_dims not in (2, 3):
        raise ValueError('image_size should has 2 or 3 elements')

    scan_num = [
        int(math.ceil(float(image_size[i]) / scan_interval[i]))
        if scan_interval[i] != 0 else 1 for i in range(num_spatial_dims)
    ]
    slices = []
    if num_spatial_dims == 3:
        for i in range(scan_num[0]):
            start_i = i * scan_interval[0]
            start_i -= max(start_i + patch_size[0] - image_size[0], 0)
            slice_i = slice(start_i, start_i + patch_size[0])

            for j in range(scan_num[1]):
                start_j = j * scan_interval[1]
                start_j -= max(start_j + patch_size[1] - image_size[1], 0)
                slice_j = slice(start_j, start_j + patch_size[1])

                for k in range(0, scan_num[2]):
                    start_k = k * scan_interval[2]
                    start_k -= max(start_k + patch_size[2] - image_size[2], 0)
                    slice_k = slice(start_k, start_k + patch_size[2])
                    slices.append((slice_i, slice_j, slice_k))
    else:
        for i in range(scan_num[0]):
            start_i = i * scan_interval[0]
            start_i -= max(start_i + patch_size[0] - image_size[0], 0)
            slice_i = slice(start_i, start_i + patch_size[0])

            for j in range(scan_num[1]):
                start_j = j * scan_interval[1]
                start_j -= max(start_j + patch_size[1] - image_size[1], 0)
                slice_j = slice(start_j, start_j + patch_size[1])
                slices.append((slice_i, slice_j))
    return slices


def sliding_window_inference(inputs, roi_size, sw_batch_size, predictor):
    """Use SlidingWindow method to execute inference.
    Args:
        inputs (torch Tensor): input image to be processed (assuming NCHW[D])
        roi_size (list, tuple): the window size to execute SlidingWindow inference.
        sw_batch_size (int): the batch size to run window slices.
        predictor (Callable): given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
    Note:
        must be channel first, support both 2D and 3D.
        input data must have batch dim.
        execute on 1 image/per inference, run a batch of window slices of 1 input image.
    """
    num_spatial_dims = len(inputs.shape) - 2
    assert len(
        roi_size
    ) == num_spatial_dims, 'roi_size {} does not match input dims.'.format(
        roi_size)

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    # TODO: Enable batch sizes > 1 in future
    if batch_size > 1:
        raise NotImplementedError

    original_image_size = [image_size[i] for i in range(num_spatial_dims)]
    # in case that image size is smaller than roi size
    image_size = tuple(
        max(image_size[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = [
        i
        for k in range(len(inputs.shape) - 1, 1, -1)
        for i in (0, max(roi_size[k - 2] - inputs.shape[k], 0))
    ]
    inputs = F.pad(inputs,
                   pad=pad_size,
                   mode='constant',
                   value=0,
                   data_format="NDHWC")

    # TODO: interval from user's specification
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)

    slice_batches = []
    for slice_index in range(0, len(slices), sw_batch_size):
        slice_index_range = range(slice_index,
                                  min(slice_index + sw_batch_size, len(slices)))
        input_slices = []
        for curr_index in slice_index_range:
            if num_spatial_dims == 3:
                slice_i, slice_j, slice_k = slices[curr_index]
                input_slices.append(inputs[0, :, slice_i, slice_j, slice_k])
            else:
                slice_i, slice_j = slices[curr_index]
                input_slices.append(inputs[0, :, slice_i, slice_j])
        slice_batches.append(torch.stack(input_slices))

    # Perform predictions
    output_rois = list()
    for data in slice_batches:
        seg_prob = predictor(data)  # batched patch segmentation

        output_rois.append(seg_prob[0].numpy())

    # stitching output image

    output_classes = output_rois[0].shape[1]
    output_shape = [batch_size, output_classes] + list(image_size)

    # allocate memory to store the full output and the count for overlapping parts
    output_image = torch.zeros(output_shape, dtype=torch.float32).numpy()
    count_map = torch.zeros(output_shape, dtype=torch.float32).numpy()

    for window_id, slice_index in enumerate(
            range(0, len(slices), sw_batch_size)):
        slice_index_range = range(slice_index,
                                  min(slice_index + sw_batch_size, len(slices)))
        # store the result in the proper location of the full output
        for curr_index in slice_index_range:
            if num_spatial_dims == 3:
                slice_i, slice_j, slice_k = slices[curr_index]
                ors = output_rois[window_id][curr_index - slice_index, :]

                output_image[0, :, slice_i, slice_j, slice_k] += ors

                count_map[0, :, slice_i, slice_j, slice_k] += 1.
            else:
                slice_i, slice_j = slices[curr_index]
                output_image[0, :, slice_i, slice_j] += output_rois[window_id][
                    curr_index - slice_index, :]
                count_map[0, :, slice_i, slice_j] += 1.

    # account for any overlapping sections
    output_image /= count_map

    output_image = paddle.to_tensor(output_image)

    if num_spatial_dims == 3:
        return (output_image[..., :original_image_size[0], :original_image_size[
            1], :original_image_size[2]], )
    return (output_image[..., :original_image_size[0], :original_image_size[1]],
            )  # 2D


def _get_scan_interval(image_size, roi_size, num_spatial_dims):
    assert (len(image_size) == num_spatial_dims
            ), 'image coord different from spatial dims.'
    assert (len(roi_size) == num_spatial_dims
            ), 'roi coord different from spatial dims.'

    scan_interval = [1 for _ in range(num_spatial_dims)]
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval[i] = int(roi_size[i])
        else:
            # this means that it's r-16 (if r>=64) and r*0.75 (if r<=64)
            scan_interval[i] = int(max(roi_size[i] - 16, roi_size[i] * 0.75))
    return tuple(scan_interval)


# todo: add aug inference with postpreocess.

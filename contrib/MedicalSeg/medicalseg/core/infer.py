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

import paddle
import paddle.nn.functional as F


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


def inference(model, im, ori_shape=None, transforms=None):
    """
    Inference for image.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        ori_shape (list): Origin shape of image.
        transforms (list): Transforms for image.

    Returns:
        Tensor: If ori_shape is not None, a prediction with shape (1, 1, d, h, w) is returned.
            If ori_shape is None, a logit with shape (1, num_classes, d, h, w) is returned.
    """
    if hasattr(model, 'data_format') and model.data_format == 'NDHWC':
        im = im.transpose((0, 2, 3, 4, 1))

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


# todo: add aug inference with postpreocess.

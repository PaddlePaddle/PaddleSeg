# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from itertools import combinations
from functools import partial

import numpy as np
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
    h, w = ori_shape[0], ori_shape[1]
    for op in transforms:
        if op.__class__.__name__ in ['Resize']:
            reverse_list.append(('resize', (h, w)))
            h, w = op.target_size[0], op.target_size[1]
        if op.__class__.__name__ in ['ResizeByLong']:
            reverse_list.append(('resize', (h, w)))
            long_edge = max(h, w)
            short_edge = min(h, w)
            short_edge = int(round(short_edge * op.long_size / long_edge))
            long_edge = op.long_size
            if h > w:
                h = long_edge
                w = short_edge
            else:
                w = long_edge
                h = short_edge
        if op.__class__.__name__ in ['Padding']:
            reverse_list.append(('padding', (h, w)))
            w, h = op.target_size[0], op.target_size[1]
        if op.__class__.__name__ in ['LimitLong']:
            long_edge = max(h, w)
            short_edge = min(h, w)
            if ((op.max_long is not None) and (long_edge > op.max_long)):
                reverse_list.append(('resize', (h, w)))
                long_edge = op.max_long
                short_edge = int(round(short_edge * op.max_long / long_edge))
            elif ((op.min_long is not None) and (long_edge < op.min_long)):
                reverse_list.append(('resize', (h, w)))
                long_edge = op.min_long
                short_edge = int(round(short_edge * op.min_long / long_edge))
            if h > w:
                h = long_edge
                w = short_edge
            else:
                w = long_edge
                h = short_edge
    return reverse_list


def reverse_transform(pred, ori_shape, transforms):
    """recover pred to origin shape"""
    reverse_list = get_reverse_list(ori_shape, transforms)
    for item in reverse_list[::-1]:
        if item[0] == 'resize':
            h, w = item[1][0], item[1][1]
            pred = F.interpolate(pred, (h, w), mode='nearest')
        elif item[0] == 'padding':
            h, w = item[1][0], item[1][1]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return pred


def find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=None):
    """
    Find the center points from the center heatmap.

    Args:
        ctr_hmp (Tensor): A Tensor of shape [1, H, W] of raw center heatmap output.
        threshold (float, optional): Threshold applied to center heatmap score. Default: 0.1.
        nms_kernel (int, optional): NMS max pooling kernel size. Default: 3.
        top_k (int, optional): An Integer, top k centers to keep. Default: None

    Returns:
        Tensor: A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    # thresholding, setting values below threshold to 0
    ctr_hmp = F.thresholded_relu(ctr_hmp, threshold)

    #NMS
    nms_padding = (nms_kernel - 1) // 2
    ctr_hmp = ctr_hmp.unsqueeze(0)
    ctr_hmp_max_pooled = F.max_pool2d(
        ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding)
    ctr_hmp = ctr_hmp * (ctr_hmp_max_pooled == ctr_hmp)

    ctr_hmp = ctr_hmp.squeeze((0, 1))
    if len(ctr_hmp.shape) != 2:
        raise ValueError('Something is wrong with center heatmap dimension.')

    if top_k is None:
        top_k_score = 0
    else:
        top_k_score, _ = paddle.topk(paddle.flatten(ctr_hmp), top_k)
        top_k_score = top_k_score[-1]
    # non-zero points are candidate centers
    ctr_hmp_k = (ctr_hmp > top_k_score[-1]).astype('int64')
    if ctr_hmp_k.sum() == 0:
        ctr_all = None
    else:
        ctr_all = paddle.nonzero(ctr_hmp_k)
    return ctr_all


def group_pixels(ctr, offsets):
    """
    Gives each pixel in the image an instance id.

    Args:
        ctr (Tensor): A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
        offsets (Tensor): A Tensor of shape [2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).

    Returns:
        Tensor: A Tensor of shape [1, H, W], ins_id is 1, 2, ...
    """
    height, width = offsets.shape[-2:]
    y_coord = paddle.arange(height, dtype=offsets.dtype).reshape([1, -1, 1])
    y_coord = paddle.concat([y_coord] * width, axis=2)
    x_coord = paddle.arange(width, dtype=offsets.dtype).reshape([1, 1, -1])
    x_coord = paddle.concat([x_coord] * height, axis=1)
    coord = paddle.concat([y_coord, x_coord], axis=0)

    ctr_loc = coord + offsets
    ctr_loc = ctr_loc.reshape((2, height * width)).transpose((1, 0))

    # ctr: [K, 2] -> [K, 1, 2]
    # ctr_loc = [H*W, 2] -> [1, H*W, 2]
    ctr = ctr.unsqueeze(1)
    ctr_loc = ctr_loc.unsqueeze(0)

    # distance: [K, H*W]
    distance = paddle.norm((ctr - ctr_loc).astype('float32'), axis=-1)

    # finds center with minimum distance at each location, offset by 1, to reserve id=0 for stuff
    instance_id = paddle.argmin(
        distance, axis=0).reshape((1, height, width)) + 1

    return instance_id


def get_instance_segmentation(semantic,
                              ctr_hmp,
                              offset,
                              thing_list,
                              threshold=0.1,
                              nms_kernel=3,
                              top_k=None):
    """
    Post-processing for instance segmentation, gets class agnostic instance id map.

    Args:
        semantic (Tensor): A Tensor of shape [1, H, W], predicted semantic label.
        ctr_hmp (Tensor): A Tensor of shape [1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        offsets (Tensor): A Tensor of shape [2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
        thing_list (list): A List of thing class id.
        threshold (float, optional): A Float, threshold applied to center heatmap score. Default: 0.1.
        nms_kernel (int, optional): An Integer, NMS max pooling kernel size. Default: 3.
        top_k (int, optional): An Integer, top k centers to keep. Default: None.

    Returns:
        Tensor: Instance segmentation results which shape is [1, H, W].
        Tensor: A Tensor of shape [1, K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    thing_seg = paddle.zeros_like(semantic)
    for thing_class in thing_list:
        thing_seg = thing_seg + (semantic == thing_class).astype('int64')
    thing_seg = (thing_seg > 0).astype('int64')
    center = find_instance_center(
        ctr_hmp, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k)
    if center is None:
        return paddle.zeros_like(semantic), center
    ins_seg = group_pixels(center, offset)
    return thing_seg * ins_seg, center.unsqueeze(0)


def merge_semantic_and_instance(semantic, instance, label_divisor, thing_list,
                                stuff_area, ignore_index):
    """
    Post-processing for panoptic segmentation, by merging semantic segmentation label and class agnostic
        instance segmentation label.

    Args:
        semantic (Tensor): A Tensor of shape [1, H, W], predicted semantic label.
        instance (Tensor): A Tensor of shape [1, H, W], predicted instance label.
        label_divisor (int): An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        thing_list (list): A List of thing class id.
        stuff_area (int): An Integer, remove stuff whose area is less tan stuff_area.
        ignore_index (int): Specifies a value that is ignored.

    Returns:
        Tensor: A Tensor of shape [1, H, W] . The pixels whose value equaling ignore_index is ignored.
            The stuff class is represented as format like class_id, while
            thing class as class_id * label_divisor + ins_id and ins_id begin from 1.
    """
    # In case thing mask does not align with semantic prediction
    pan_seg = paddle.zeros_like(semantic) + ignore_index
    thing_seg = instance > 0
    semantic_thing_seg = paddle.zeros_like(semantic)
    for thing_class in thing_list:
        semantic_thing_seg += semantic == thing_class

    # keep track of instance id for each class
    class_id_tracker = {}

    # paste thing by majority voting
    ins_ids = paddle.unique(instance)
    for ins_id in ins_ids:
        if ins_id == 0:
            continue
        # Make sure only do majority voting within semantic_thing_seg
        thing_mask = paddle.logical_and(instance == ins_id,
                                        semantic_thing_seg == 1)
        if paddle.all(paddle.logical_not(thing_mask)):
            continue
        # get class id for instance of ins_id
        sem_ins_id = paddle.gather(
            semantic.reshape((-1, )),
            paddle.nonzero(thing_mask.reshape((
                -1, ))))  # equal to semantic[thing_mask]
        v, c = paddle.unique(sem_ins_id, return_counts=True)
        class_id = paddle.gather(v, c.argmax())
        class_id = class_id.numpy()[0]
        if class_id in class_id_tracker:
            new_ins_id = class_id_tracker[class_id]
        else:
            class_id_tracker[class_id] = 1
            new_ins_id = 1
        class_id_tracker[class_id] += 1

        # pan_seg[thing_mask] = class_id * label_divisor + new_ins_id
        pan_seg = pan_seg * (paddle.logical_not(thing_mask)) + (
            class_id * label_divisor + new_ins_id) * thing_mask.astype('int64')

    # paste stuff to unoccupied area
    class_ids = paddle.unique(semantic)
    for class_id in class_ids:
        if class_id.numpy() in thing_list:
            # thing class
            continue
        # calculate stuff area
        stuff_mask = paddle.logical_and(semantic == class_id,
                                        paddle.logical_not(thing_seg))
        area = paddle.sum(stuff_mask.astype('int64'))
        if area >= stuff_area:
            # pan_seg[stuff_mask] = class_id
            pan_seg = pan_seg * (paddle.logical_not(stuff_mask)
                                 ) + stuff_mask.astype('int64') * class_id

    return pan_seg


def inference(
        model,
        im,
        transforms,
        thing_list,
        label_divisor,
        stuff_area,
        ignore_index,
        threshold=0.1,
        nms_kernel=3,
        top_k=None,
        ori_shape=None, ):
    """
    Inference for image.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        transforms (list): Transforms for image.
        thing_list (list): A List of thing class id.
        label_divisor (int): An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
        stuff_area (int): An Integer, remove stuff whose area is less tan stuff_area.
        ignore_index (int): Specifies a value that is ignored.
        threshold (float, optional): A Float, threshold applied to center heatmap score. Default: 0.1.
        nms_kernel (int, optional): An Integer, NMS max pooling kernel size. Default: 3.
        top_k (int, optional): An Integer, top k centers to keep. Default: None.
        ori_shape (list. optional): Origin shape of image. Default: None.

    Returns:
        list: A list of [semantic, semantic_softmax, instance, panoptic, ctr_hmp].
            semantic: Semantic segmentation results with shape [1, 1, H, W], which value is 0, 1, 2...
            semantic_softmax: A Tensor represent probabilities for each class, which shape is [1, num_classes, H, W].
            instance: Instance segmentation results with class agnostic, which value is 0, 1, 2, ..., and 0 is stuff.
            panoptic: Panoptic segmentation results which value is ignore_index, stuff_id, thing_id * label_divisor + ins_id , ins_id >= 1.
    """
    logits = model(im)
    # semantic: [1, c, h, w], center: [1, 1, h, w], offset: [1, 2, h, w]
    semantic, ctr_hmp, offset = logits
    semantic = paddle.argmax(semantic, axis=1, keepdim=True)
    semantic = semantic.squeeze(0)  # shape: [1, h, w]
    semantic_softmax = F.softmax(logits[0], axis=1).squeeze()
    ctr_hmp = ctr_hmp.squeeze(0)  # shape: [1, h, w]
    offset = offset.squeeze(0)  # shape: [2, h, w]

    instance, center = get_instance_segmentation(
        semantic=semantic,
        ctr_hmp=ctr_hmp,
        offset=offset,
        thing_list=thing_list,
        threshold=threshold,
        nms_kernel=nms_kernel,
        top_k=top_k)
    panoptic = merge_semantic_and_instance(semantic, instance, label_divisor,
                                           thing_list, stuff_area, ignore_index)

    # Recover to origin shape
    # semantic: 0, 1, 2, 3...
    # instance: 0, 1, 2, 3, 4,  5... and the 0 is stuff.
    # panoptic: ignore_index, stuff_id, thing_id * label_divisor + ins_id , ins_id >= 1.
    results = [semantic, semantic_softmax, instance, panoptic, ctr_hmp]
    if ori_shape is not None:
        results = [i.unsqueeze(0) for i in results]
        results = [
            reverse_transform(
                i, ori_shape=ori_shape, transforms=transforms) for i in results
        ]

    return results

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import os.path as osp

import numpy as np
from PIL import Image
import time
import paddle
import paddle.nn.functional as F

from paddleseg.utils import metrics, TimeAverager, calculate_eta, logger, progbar, get_image_list
from paddleseg.core.infer import reverse_transform


def evaluate(
        img_dir,
        gt_dir,
        num_classes,
        ignore_index=255,
        print_detail=True, ):
    """
    Launch evalution.

    Args:
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to evaluate by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0

    image_list, _ = get_image_list(img_dir)
    for im_path in image_list:
        im_name = osp.basename(im_path)
        gt_path = osp.join(gt_dir, im_name)
        img = paddle.to_tensor(np.asarray(Image.open(im_path)), dtype='int32')
        gt = paddle.to_tensor(np.asarray(Image.open(gt_path)), dtype='int32')
        img = paddle.unsqueeze(img, axis=[0, 1])
        gt = paddle.unsqueeze(gt, axis=[0, 1])

        ori_shape = gt.shape[-2:]
        pred = F.interpolate(img, ori_shape, mode='nearest')

        intersect_area, pred_area, label_area = metrics.calculate_area(
            pred, gt, num_classes, ignore_index=ignore_index)

        intersect_area_all = intersect_area_all + intersect_area
        pred_area_all = pred_area_all + pred_area
        label_area_all = label_area_all + label_area

    class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                                       label_area_all)
    class_acc, acc = metrics.accuracy(intersect_area_all, pred_area_all)
    kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)

    if print_detail:
        print("#Images: {} \nmIoU: {} \nAcc: {} \nKappa: {} ".format(
            len(image_list), miou, acc, kappa))
        print("Class IoU: " + str(class_iou))
        print("Class Acc: " + str(class_acc))
    metrics_dict = {}
    metrics_dict['#Images'] = len(image_list)
    metrics_dict['mIoU'] = miou
    metrics_dict['Acc'] = acc
    metrics_dict['Kappa'] = kappa
    metrics_dict['Class IoU'] = class_iou
    metrics_dict['Class Acc'] = class_acc
    return metrics_dict

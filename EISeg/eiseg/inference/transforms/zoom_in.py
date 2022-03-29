# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is based on https://github.com/saic-vul/ritm_interactive_segmentation
Ths copyright of saic-vul/ritm_interactive_segmentation is as follows:
MIT License [see LICENSE for details]
"""

import paddle
import numpy as np
from inference.clicker import Click
from util.misc import get_bbox_iou, get_bbox_from_mask, expand_bbox, clamp_bbox
from .base import BaseTransform


class ZoomIn(BaseTransform):
    def __init__(
            self,
            target_size=700,
            skip_clicks=1,
            expansion_ratio=1.4,
            min_crop_size=480,
            recompute_thresh_iou=0.5,
            prob_thresh=0.50, ):
        super().__init__()
        self.target_size = target_size
        self.min_crop_size = min_crop_size
        self.skip_clicks = skip_clicks
        self.expansion_ratio = expansion_ratio
        self.recompute_thresh_iou = recompute_thresh_iou
        self.prob_thresh = prob_thresh

        self._input_image_shape = None
        self._prev_probs = None
        self._object_roi = None
        self._roi_image = None

    def transform(self, image_nd, clicks_lists):
        assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
        self.image_changed = False

        clicks_list = clicks_lists[0]
        if len(clicks_list) <= self.skip_clicks:
            return image_nd, clicks_lists

        self._input_image_shape = image_nd.shape

        current_object_roi = None
        if self._prev_probs is not None:
            current_pred_mask = (self._prev_probs > self.prob_thresh)[0, 0]
            if current_pred_mask.sum() > 0:
                current_object_roi = get_object_roi(
                    current_pred_mask,
                    clicks_list,
                    self.expansion_ratio,
                    self.min_crop_size, )

        if current_object_roi is None:
            if self.skip_clicks >= 0:
                return image_nd, clicks_lists
            else:
                current_object_roi = 0, image_nd.shape[
                    2] - 1, 0, image_nd.shape[3] - 1

        update_object_roi = False
        if self._object_roi is None:
            update_object_roi = True
        elif not check_object_roi(self._object_roi, clicks_list):
            update_object_roi = True
        elif (get_bbox_iou(current_object_roi, self._object_roi) <
              self.recompute_thresh_iou):
            update_object_roi = True

        if update_object_roi:
            self._object_roi = current_object_roi
            self.image_changed = True
        self._roi_image = get_roi_image_nd(image_nd, self._object_roi,
                                           self.target_size)

        tclicks_lists = [self._transform_clicks(clicks_list)]
        return self._roi_image, tclicks_lists

    def inv_transform(self, prob_map):
        if self._object_roi is None:
            self._prev_probs = prob_map.numpy()
            return prob_map

        assert prob_map.shape[0] == 1
        rmin, rmax, cmin, cmax = self._object_roi
        prob_map = paddle.nn.functional.interpolate(
            prob_map,
            size=(rmax - rmin + 1, cmax - cmin + 1),
            mode="bilinear",
            align_corners=True, )

        if self._prev_probs is not None:
            new_prob_map = paddle.zeros(
                shape=self._prev_probs.shape, dtype=prob_map.dtype)
            new_prob_map[:, :, rmin:rmax + 1, cmin:cmax + 1] = prob_map
        else:
            new_prob_map = prob_map

        self._prev_probs = new_prob_map.numpy()

        return new_prob_map

    def check_possible_recalculation(self):
        if (self._prev_probs is None or self._object_roi is not None or
                self.skip_clicks > 0):
            return False

        pred_mask = (self._prev_probs > self.prob_thresh)[0, 0]
        if pred_mask.sum() > 0:
            possible_object_roi = get_object_roi(
                pred_mask, [], self.expansion_ratio, self.min_crop_size)
            image_roi = (
                0,
                self._input_image_shape[2] - 1,
                0,
                self._input_image_shape[3] - 1, )
            if get_bbox_iou(possible_object_roi, image_roi) < 0.50:
                return True
        return False

    def get_state(self):
        roi_image = self._roi_image if self._roi_image is not None else None
        return (
            self._input_image_shape,
            self._object_roi,
            self._prev_probs,
            roi_image,
            self.image_changed, )

    def set_state(self, state):
        (
            self._input_image_shape,
            self._object_roi,
            self._prev_probs,
            self._roi_image,
            self.image_changed, ) = state

    def reset(self):
        self._input_image_shape = None
        self._object_roi = None
        self._prev_probs = None
        self._roi_image = None
        self.image_changed = False

    def _transform_clicks(self, clicks_list):
        if self._object_roi is None:
            return clicks_list

        rmin, rmax, cmin, cmax = self._object_roi
        crop_height, crop_width = self._roi_image.shape[2:]

        transformed_clicks = []
        for click in clicks_list:
            new_r = crop_height * (click.coords[0] - rmin) / (rmax - rmin + 1)
            new_c = crop_width * (click.coords[1] - cmin) / (cmax - cmin + 1)
            transformed_clicks.append(click.copy(coords=(new_r, new_c)))
        return transformed_clicks


def get_object_roi(pred_mask, clicks_list, expansion_ratio, min_crop_size):
    pred_mask = pred_mask.copy()

    for click in clicks_list:
        if click.is_positive:
            pred_mask[int(click.coords[0]), int(click.coords[1])] = 1

    bbox = get_bbox_from_mask(pred_mask)
    bbox = expand_bbox(bbox, expansion_ratio, min_crop_size)
    h, w = pred_mask.shape[0], pred_mask.shape[1]
    bbox = clamp_bbox(bbox, 0, h - 1, 0, w - 1)

    return bbox


def get_roi_image_nd(image_nd, object_roi, target_size):
    rmin, rmax, cmin, cmax = object_roi

    height = rmax - rmin + 1
    width = cmax - cmin + 1

    if isinstance(target_size, tuple):
        new_height, new_width = target_size
    else:
        scale = target_size / max(height, width)
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))

    with paddle.no_grad():
        roi_image_nd = image_nd[:, :, rmin:rmax + 1, cmin:cmax + 1]
        roi_image_nd = paddle.nn.functional.interpolate(
            roi_image_nd,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=True, )

    return roi_image_nd


def check_object_roi(object_roi, clicks_list):
    for click in clicks_list:
        if click.is_positive:
            if click.coords[0] < object_roi[0] or click.coords[0] >= object_roi[
                    1]:
                return False
            if click.coords[1] < object_roi[2] or click.coords[1] >= object_roi[
                    3]:
                return False

    return True

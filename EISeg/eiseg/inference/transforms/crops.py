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


import math

import paddle
import numpy as np

from inference.clicker import Click
from .base import BaseTransform


class Crops(BaseTransform):
    def __init__(self, crop_size=(320, 480), min_overlap=0.2):
        super().__init__()
        self.crop_height, self.crop_width = crop_size
        self.min_overlap = min_overlap

        self.x_offsets = None
        self.y_offsets = None
        self._counts = None

    def transform(self, image_nd, clicks_lists):
        assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
        image_height, image_width = image_nd.shape[2:4]
        self._counts = None

        if image_height < self.crop_height or image_width < self.crop_width:
            return image_nd, clicks_lists

        self.x_offsets = get_offsets(image_width, self.crop_width, self.min_overlap)
        self.y_offsets = get_offsets(image_height, self.crop_height, self.min_overlap)
        self._counts = np.zeros((image_height, image_width))

        image_crops = []
        for dy in self.y_offsets:
            for dx in self.x_offsets:
                self._counts[dy : dy + self.crop_height, dx : dx + self.crop_width] += 1
                image_crop = image_nd[
                    :, :, dy : dy + self.crop_height, dx : dx + self.crop_width
                ]
                image_crops.append(image_crop)
        image_crops = paddle.concat(image_crops, axis=0)
        self._counts = paddle.to_tensor(self._counts, dtype="float32")

        clicks_list = clicks_lists[0]
        clicks_lists = []
        for dy in self.y_offsets:
            for dx in self.x_offsets:
                crop_clicks = [
                    x.copy(coords=(x.coords[0] - dy, x.coords[1] - dx))
                    for x in clicks_list
                ]
                clicks_lists.append(crop_clicks)

        return image_crops, clicks_lists

    def inv_transform(self, prob_map):
        if self._counts is None:
            return prob_map

        new_prob_map = paddle.zeros((1, 1, *self._counts.shape), dtype=prob_map.dtype)

        crop_indx = 0
        for dy in self.y_offsets:
            for dx in self.x_offsets:
                new_prob_map[
                    0, 0, dy : dy + self.crop_height, dx : dx + self.crop_width
                ] += prob_map[crop_indx, 0]
                crop_indx += 1
        new_prob_map = paddle.divide(new_prob_map, self._counts)

        return new_prob_map

    def get_state(self):
        return self.x_offsets, self.y_offsets, self._counts

    def set_state(self, state):
        self.x_offsets, self.y_offsets, self._counts = state

    def reset(self):
        self.x_offsets = None
        self.y_offsets = None
        self._counts = None


def get_offsets(length, crop_size, min_overlap_ratio=0.2):
    if length == crop_size:
        return [0]

    N = (length / crop_size - min_overlap_ratio) / (1 - min_overlap_ratio)
    N = math.ceil(N)

    overlap_ratio = (N - length / crop_size) / (N - 1)
    overlap_width = int(crop_size * overlap_ratio)

    offsets = [0]
    for i in range(1, N):
        new_offset = offsets[-1] + crop_size - overlap_width
        if new_offset + crop_size > length:
            new_offset = length - crop_size

        offsets.append(new_offset)

    return offsets

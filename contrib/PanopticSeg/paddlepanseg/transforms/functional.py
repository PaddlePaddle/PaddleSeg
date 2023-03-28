# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np


def rgb2id(color):
    if color.dtype == np.uint8:
        color = color.astype(np.int32)
    return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]


def id2rgb(id_map):
    id_map_copy = id_map.copy()
    rgb_shape = tuple(list(id_map.shape) + [3])
    rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
    for i in range(3):
        rgb_map[..., i] = id_map_copy % 256
        id_map_copy //= 256
    return rgb_map

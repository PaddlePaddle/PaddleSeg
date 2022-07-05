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

import numpy as np
import cv2
import operator
from functools import reduce


# 2%线性拉伸
def two_percentLinear(image: np.array, max_out: int=255,
                      min_out: int=0) -> np.array:
    b, g, r = cv2.split(image)

    def __gray_process(gray, maxout=max_out, minout=min_out):
        high_value = np.percentile(gray, 98)  # 取得98%直方图处对应灰度
        low_value = np.percentile(gray, 2)
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
        processed_gray = ((truncated_gray - low_value) /
                          (high_value - low_value)) * (maxout - minout)
        return processed_gray

    r_p = __gray_process(r)
    g_p = __gray_process(g)
    b_p = __gray_process(b)
    result = cv2.merge((b_p, g_p, r_p))
    return np.uint8(result)


# 简单图像标准化
def sample_norm(image: np.array, NUMS: int=65536) -> np.array:
    if NUMS == 256:
        return np.uint8(image)
    stretched_r = __stretch(image[:, :, 0], NUMS)
    stretched_g = __stretch(image[:, :, 1], NUMS)
    stretched_b = __stretch(image[:, :, 2], NUMS)
    stretched_img = cv2.merge([
        stretched_r / float(NUMS),
        stretched_g / float(NUMS),
        stretched_b / float(NUMS),
    ])
    return np.uint8(stretched_img * 255)


# 直方图均衡化
def __stretch(ima: np.array, NUMS: int) -> np.array:
    hist = __histogram(ima, NUMS)
    lut = []
    for bt in range(0, len(hist), NUMS):
        # 步长尺寸
        step = reduce(operator.add, hist[bt:bt + NUMS]) / (NUMS - 1)
        # 创建均衡的查找表
        n = 0
        for i in range(NUMS):
            lut.append(n / step)
            n += hist[i + bt]
        np.take(lut, ima, out=ima)
        return ima


# 计算直方图
def __histogram(ima: np.array, NUMS: int) -> np.array:
    bins = list(range(0, NUMS))
    flat = ima.flat
    n = np.searchsorted(np.sort(flat), bins)
    n = np.concatenate([n, [len(flat)]])
    hist = n[1:] - n[:-1]
    return hist


# 计算缩略图
def get_thumbnail(image: np.array, range: int=2000,
                  max_size: int=1000) -> np.array:
    h, w = image.shape[:2]
    if h >= range or w >= range:
        if h >= w:
            image = cv2.resize(image, (int(max_size / h * w), max_size))
        else:
            image = cv2.resize(image, (max_size, int(max_size / w * h)))
    return image

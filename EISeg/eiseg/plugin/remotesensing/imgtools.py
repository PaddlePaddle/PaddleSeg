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
from skimage import exposure


# 2%线性拉伸
def two_percentLinear(image: np.ndarray, 
                      max_out: int=255, 
                      min_out: int=0) -> np.ndarray:
    b, g, r = cv2.split(image)

    def __gray_process(gray, maxout=max_out, minout=min_out):
        high_value = np.percentile(gray, 98)  # 取得98%直方图处对应灰度
        low_value = np.percentile(gray, 2)
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
        processed_gray = ((truncated_gray - low_value) / (high_value - low_value)) * (
            maxout - minout)
        return processed_gray

    r_p = __gray_process(r)
    g_p = __gray_process(g)
    b_p = __gray_process(b)
    result = cv2.merge((b_p, g_p, r_p))
    return np.uint8(result)


# 简单图像标准化
def sample_norm(image: np.ndarray) -> np.ndarray:
    stretches = []
    if len(image.shape) == 3:
        for b in range(image.shape[-1]):
            stretched = exposure.equalize_hist(image[:, :, b])
            stretched /= float(np.max(stretched))
            stretches.append(stretched)
        stretched_img = np.stack(stretches, axis=2)
    else:  # if len(image.shape) == 2
        stretched_img = exposure.equalize_hist(image)
    return np.uint8(stretched_img * 255)


# 计算缩略图
def get_thumbnail(image: np.ndarray, 
                  range: int=2000, 
                  max_size: int=1000) -> np.ndarray:
    h, w = image.shape[:2]
    if h >= range or w >= range:
        if h >= w:
            image = cv2.resize(image, (int(max_size / h * w), max_size))
        else:
            image = cv2.resize(image, (max_size, int(max_size / w * h)))
    return image

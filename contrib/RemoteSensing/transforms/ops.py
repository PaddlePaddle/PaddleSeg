# coding: utf8
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

import cv2
import math
import numpy as np
from PIL import Image, ImageEnhance


def normalize(im, min_value, max_value, mean, std):
    # Rescaling (min-max normalization)
    range_value = [max_value[i] - min_value[i] for i in range(len(max_value))]
    im = (im.astype(np.float32, copy=False) - min_value) / range_value

    # Standardization (Z-score Normalization)
    im -= mean
    im /= std
    return im


def permute(im, to_bgr=False):
    im = np.swapaxes(im, 1, 2)
    im = np.swapaxes(im, 1, 0)
    if to_bgr:
        im = im[[2, 1, 0], :, :]
    return im


def _resize(im, shape):
    return cv2.resize(im, shape)


def resize_short(im, short_size=224):
    percent = float(short_size) / min(im.shape[0], im.shape[1])
    resized_width = int(round(im.shape[1] * percent))
    resized_height = int(round(im.shape[0] * percent))
    im = _resize(im, shape=(resized_width, resized_height))
    return im


def resize_long(im, long_size=224, interpolation=cv2.INTER_LINEAR):
    value = max(im.shape[0], im.shape[1])
    scale = float(long_size) / float(value)
    im = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=interpolation)
    return im


def random_crop(im,
                crop_size=224,
                lower_scale=0.08,
                lower_ratio=3. / 4,
                upper_ratio=4. / 3):
    scale = [lower_scale, 1.0]
    ratio = [lower_ratio, upper_ratio]
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio
    bound = min((float(im.shape[0]) / im.shape[1]) / (h**2),
                (float(im.shape[1]) / im.shape[0]) / (w**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)
    target_area = im.shape[0] * im.shape[1] * np.random.uniform(
        scale_min, scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)
    i = np.random.randint(0, im.shape[0] - h + 1)
    j = np.random.randint(0, im.shape[1] - w + 1)
    im = im[i:i + h, j:j + w, :]
    im = _resize(im, shape=(crop_size, crop_size))
    return im


def center_crop(im, crop_size=224):
    height, width = im.shape[:2]
    w_start = (width - crop_size) // 2
    h_start = (height - crop_size) // 2
    w_end = w_start + crop_size
    h_end = h_start + crop_size
    im = im[h_start:h_end, w_start:w_end, :]
    return im


def horizontal_flip(im):
    if len(im.shape) == 3:
        im = im[:, ::-1, :]
    elif len(im.shape) == 2:
        im = im[:, ::-1]
    return im


def vertical_flip(im):
    if len(im.shape) == 3:
        im = im[::-1, :, :]
    elif len(im.shape) == 2:
        im = im[::-1, :]
    return im


def bgr2rgb(im):
    return im[:, :, ::-1]


def brightness(im, brightness_lower, brightness_upper):
    brightness_delta = np.random.uniform(brightness_lower, brightness_upper)
    im = ImageEnhance.Brightness(im).enhance(brightness_delta)
    return im


def contrast(im, contrast_lower, contrast_upper):
    contrast_delta = np.random.uniform(contrast_lower, contrast_upper)
    im = ImageEnhance.Contrast(im).enhance(contrast_delta)
    return im


def saturation(im, saturation_lower, saturation_upper):
    saturation_delta = np.random.uniform(saturation_lower, saturation_upper)
    im = ImageEnhance.Color(im).enhance(saturation_delta)
    return im


def hue(im, hue_lower, hue_upper):
    hue_delta = np.random.uniform(hue_lower, hue_upper)
    im = np.array(im.convert('HSV'))
    im[:, :, 0] = im[:, :, 0] + hue_delta
    im = Image.fromarray(im, mode='HSV').convert('RGB')
    return im


def rotate(im, rotate_lower, rotate_upper):
    rotate_delta = np.random.uniform(rotate_lower, rotate_upper)
    im = im.rotate(int(rotate_delta))
    return im


def resize_padding(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(
            max_side_len) / resize_h if resize_h > resize_w else float(
                max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    #im = cv2.resize(im, (512, 512))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    _ratio = np.array([ratio_h, ratio_w]).reshape(-1, 2)
    return im, _ratio

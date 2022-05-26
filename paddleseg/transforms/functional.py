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
import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage.morphology import distance_transform_edt


def normalize(im, mean, std):
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im


def resize(im, target_size=608, interp=cv2.INTER_LINEAR):
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        w = target_size[0]
        h = target_size[1]
    else:
        w = target_size
        h = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


def resize_long(im, long_size=224, interpolation=cv2.INTER_LINEAR):
    value = max(im.shape[0], im.shape[1])
    scale = float(long_size) / float(value)
    resized_width = int(round(im.shape[1] * scale))
    resized_height = int(round(im.shape[0] * scale))

    im = cv2.resize(
        im, (resized_width, resized_height), interpolation=interpolation)
    return im


def resize_short(im, short_size=224, interpolation=cv2.INTER_LINEAR):
    value = min(im.shape[0], im.shape[1])
    scale = float(short_size) / float(value)
    resized_width = int(round(im.shape[1] * scale))
    resized_height = int(round(im.shape[0] * scale))

    im = cv2.resize(
        im, (resized_width, resized_height), interpolation=interpolation)
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


def sharpness(im, sharpness_lower, sharpness_upper):
    sharpness_delta = np.random.uniform(sharpness_lower, sharpness_upper)
    im = ImageEnhance.Sharpness(im).enhance(sharpness_delta)
    return im


def rotate(im, rotate_lower, rotate_upper):
    rotate_delta = np.random.uniform(rotate_lower, rotate_upper)
    im = im.rotate(int(rotate_delta))
    return im


def mask_to_onehot(mask, num_classes):
    """
    Convert a mask (H, W) to onehot (K, H, W).

    Args:
        mask (np.ndarray): Label mask with shape (H, W)
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Onehot mask with shape(K, H, W).
    """
    _mask = [mask == i for i in range(num_classes)]
    _mask = np.array(_mask).astype(np.uint8)
    return _mask


def onehot_to_binary_edge(mask, radius):
    """
    Convert a onehot mask (K, H, W) to a edge mask.

    Args:
        mask (np.ndarray): Onehot mask with shape (K, H, W)
        radius (int|float): Radius of edge.

    Returns:
        np.ndarray: Edge mask with shape(H, W).
    """
    if radius < 1:
        raise ValueError('`radius` should be greater than or equal to 1')
    num_classes = mask.shape[0]

    edge = np.zeros(mask.shape[1:])
    # pad borders
    mask = np.pad(mask, ((0, 0), (1, 1), (1, 1)),
                  mode='constant',
                  constant_values=0)
    for i in range(num_classes):
        dist = distance_transform_edt(mask[i, :]) + distance_transform_edt(
            1.0 - mask[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edge += dist

    edge = np.expand_dims(edge, axis=0)
    edge = (edge > 0).astype(np.uint8)
    return edge


def mask_to_binary_edge(mask, radius, num_classes):
    """
    Convert a segmentic segmentation mask (H, W) to a binary edge mask(H, W).

    Args:
        mask (np.ndarray): Label mask with shape (H, W)
        radius (int|float): Radius of edge.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Edge mask with shape(H, W).
    """
    mask = mask.squeeze()
    onehot = mask_to_onehot(mask, num_classes)
    edge = onehot_to_binary_edge(onehot, radius)
    return edge


def add_margin(pil_img, top, right, bottom, left, margin_color):
    """
    Add margin around an image
    top, right, bottom, left are the margin widths, in pixels
    margin_color is what to use for the margins
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), margin_color)
    result.paste(pil_img, (left, top))
    return result


def set_crop_size(crop_size):
    if isinstance(crop_size, (list, tuple)):
        size = crop_size
    elif isinstance(crop_size, numbers.Number):
        size = (int(crop_size), int(crop_size))
    else:
        raise
    return size


class RandomCrop(object):
    """
    Take a random crop from the image.
    First the image or crop size may need to be adjusted if the incoming image
    is too small...
    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image
    A random crop is taken such that the crop fits within the image.
    if cfg.DATASET.TRANSLATION_AUG_FIX is set, we insure that there's always
    translation randomness of at least that value around the image.
    if image < crop_size:
        # slide crop within image, random offset
    else:
        # slide image within crop
    """

    def __init__(self, crop_size, nopad=True):
        self.size = set_crop_size(crop_size)
        #self.ignore_index = cfg.DATASET.IGNORE_LABEL
        self.ignore_index = 255
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    @staticmethod
    def crop_in_image(centroid, target_w, target_h, w, h, img, mask):
        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - target_w
            max_y = h - target_h
            x1 = random.randint(c_x - target_w, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - target_h, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == target_w:
                x1 = 0
            else:
                x1 = random.randint(0, w - target_w)
            if h == target_h:
                y1 = 0
            else:
                y1 = random.randint(0, h - target_h)

        return [
            img.crop((x1, y1, x1 + target_w, y1 + target_h)), mask.crop(
                (x1, y1, x1 + target_w, y1 + target_h))
        ]

    def image_in_crop(self, target_w, target_h, w, h, img, mask):
        # image smaller than crop, so slide image within crop
        x_total_margin = target_w - w
        y_total_margin = target_h - h

        left = random.randint(0, x_total_margin)
        right = x_total_margin - left

        top = random.randint(0, y_total_margin)
        bottom = y_total_margin - top

        slid_image = add_margin(img, top, right, bottom, left, self.pad_color)
        slid_mask = add_margin(mask, top, right, bottom, left,
                               self.ignore_index)
        return [slid_image, slid_mask]

    def __call__(self, img, mask, centroid=None):
        assert img.size == mask.size
        w, h = img.size
        target_h, target_w = self.size  # ASSUME H, W

        if w == target_w and h == target_h:
            return [img, mask]

        #if cfg.DATASET.TRANSLATE_AUG_FIX: #False
        if False:
            if w < target_w and h < target_h:
                return self.image_in_crop(target_w, target_h, w, h, img, mask)
            else:
                return self.crop_in_image(centroid, target_w, target_h, w, h,
                                          img, mask)

        if self.nopad:
            # Shrink crop size if image < crop
            if target_h > h or target_w > w:
                shorter_side = min(w, h)
                target_h, target_w = shorter_side, shorter_side
        else:
            # Pad image if image < crop
            if target_h > h:
                pad_h = (target_h - h) // 2 + 1
            else:
                pad_h = 0
            if target_w > w:
                pad_w = (target_w - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                mask = ImageOps.expand(
                    mask, border=border, fill=self.ignore_index)
                w, h = img.size

        return self.crop_in_image(centroid, target_w, target_h, w, h, img, mask)

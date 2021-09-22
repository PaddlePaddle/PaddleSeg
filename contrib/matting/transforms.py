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

import random

import cv2
import numpy as np
from paddleseg.transforms import functional


class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].
    """

    def __init__(self, transforms, to_rgb=True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.to_rgb = to_rgb

    def __call__(self, data):
        """
        Args:
            data (dict): The data to transform.

        Returns:
            dict: Data after transformation
        """
        for op in self.transforms:
            data = op(data)
            if data is None:
                return None

        data['img'] = np.transpose(data['img'], (2, 0, 1))
        for key in data.get('gt_fields', []):
            if len(data[key].shape) == 2:
                continue
            data[key] = np.transpose(data[key], (2, 0, 1))

        return data


class LoadImages:
    def __init__(self, to_rgb=True):
        self.to_rgb = to_rgb

    def __call__(self, data):
        if isinstance(data['img'], str):
            data['img'] = cv2.imread(data['img'])
        for key in data.get('gt_fields', []):
            if isinstance(data[key], str):
                data[key] = cv2.imread(data[key], cv2.IMREAD_UNCHANGED)
            # if alpha and trimap has 3 channels, extract one.
            if key in ['alpha', 'trimap']:
                if len(data[key].shape) > 2:
                    data[key] = data[key][:, :, 0]

        if self.to_rgb:
            data['img'] = cv2.cvtColor(data['img'], cv2.COLOR_BGR2RGB)
            for key in data.get('gt_fields', []):
                if len(data[key].shape) == 2:
                    continue
                data[key] = cv2.cvtColor(data[key], cv2.COLOR_BGR2RGB)

        return data


class Resize:
    def __init__(self, target_size=(512, 512)):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, data):
        data['trans_info'].append(('resize', data['img'].shape[0:2]))
        data['img'] = functional.resize(data['img'], self.target_size)
        for key in data.get('gt_fields', []):
            data[key] = functional.resize(data[key], self.target_size)
        return data


class ResizeByLong:
    """
    Resize the long side of an image to given size, and then scale the other side proportionally.

    Args:
        long_size (int): The target size of long side.
    """

    def __init__(self, long_size):
        self.long_size = long_size

    def __call__(self, data):
        data['trans_info'].append(('resize', data['img'].shape[0:2]))
        data['img'] = functional.resize_long(data['img'], self.long_size)
        for key in data.get('gt_fields', []):
            data[key] = functional.resize_long(data[key], self.long_size)
        return data


class ResizeByShort:
    """
    Resize the short side of an image to given size, and then scale the other side proportionally.

    Args:
        short_size (int): The target size of short side.
    """

    def __init__(self, short_size):
        self.short_size = short_size

    def __call__(self, data):
        data['trans_info'].append(('resize', data['img'].shape[0:2]))
        data['img'] = functional.resize_short(data['img'], self.short_size)
        for key in data.get('gt_fields', []):
            data[key] = functional.resize_short(data[key], self.short_size)
        return data


class ResizeToIntMult:
    """
    Resize to some int muitple, d.g. 32.
    """

    def __init__(self, mult_int=32):
        self.mult_int = mult_int

    def __call__(self, data):
        data['trans_info'].append(('resize', data['img'].shape[0:2]))

        h, w = data['img'].shape[0:2]
        rw = w - w % 32
        rh = h - h % 32
        data['img'] = functional.resize(data['img'], (rw, rh))
        for key in data.get('gt_fields', []):
            data[key] = functional.resize(data[key], (rw, rh))

        return data


class Normalize:
    """
    Normalize an image.

    Args:
        mean (list, optional): The mean value of a data set. Default: [0.5, 0.5, 0.5].
        std (list, optional): The standard deviation of a data set. Default: [0.5, 0.5, 0.5].

    Raises:
        ValueError: When mean/std is not list or any value in std is 0.
    """

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, (list, tuple))
                and isinstance(self.std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, data):
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        data['img'] = functional.normalize(data['img'], mean, std)
        if 'fg' in data.get('gt_fields', []):
            data['fg'] = functional.normalize(data['fg'], mean, std)
        if 'bg' in data.get('gt_fields', []):
            data['bg'] = functional.normalize(data['bg'], mean, std)

        return data


class RandomCropByAlpha:
    """
    Randomly crop while centered on uncertain area by a certain probability.

    Args:
        crop_size (tuple|list): The size you want to crop from image.
        p (float): The probability centered on uncertain area.

    """

    def __init__(self, crop_size=((320, 320), (480, 480), (640, 640)),
                 prob=0.5):
        self.crop_size = crop_size
        self.prob = prob

    def __call__(self, data):
        idex = np.random.randint(low=0, high=len(self.crop_size))
        crop_w, crop_h = self.crop_size[idex]

        img_h = data['img'].shape[0]
        img_w = data['img'].shape[1]
        if np.random.rand() < self.prob:
            crop_center = np.where((data['alpha'] > 0) & (data['alpha'] < 255))
            center_h_array, center_w_array = crop_center
            if len(center_h_array) == 0:
                return data
            rand_ind = np.random.randint(len(center_h_array))
            center_h = center_h_array[rand_ind]
            center_w = center_w_array[rand_ind]
            delta_h = crop_h // 2
            delta_w = crop_w // 2
            start_h = max(0, center_h - delta_h)
            start_w = max(0, center_w - delta_w)
        else:
            start_h = 0
            start_w = 0
            if img_h > crop_h:
                start_h = np.random.randint(img_h - crop_h + 1)
            if img_w > crop_w:
                start_w = np.random.randint(img_w - crop_w + 1)

        end_h = min(img_h, start_h + crop_h)
        end_w = min(img_w, start_w + crop_w)

        data['img'] = data['img'][start_h:end_h, start_w:end_w]
        for key in data.get('gt_fields', []):
            data[key] = data[key][start_h:end_h, start_w:end_w]

        return data


class RandomCrop:
    """
    Randomly crop

    Args:
    crop_size (tuple|list): The size you want to crop from image.
    """

    def __init__(self, crop_size=((320, 320), (480, 480), (640, 640))):
        self.crop_size = crop_size

    def __call__(self, data):
        idex = np.random.randint(low=0, high=len(self.crop_size))
        crop_w, crop_h = self.crop_size[idex]
        img_h, img_w = data['img'].shape[0:2]

        start_h = 0
        start_w = 0
        if img_h > crop_h:
            start_h = np.random.randint(img_h - crop_h + 1)
        if img_w > crop_w:
            start_w = np.random.randint(img_w - crop_w + 1)

        end_h = min(img_h, start_h + crop_h)
        end_w = min(img_w, start_w + crop_w)

        data['img'] = data['img'][start_h:end_h, start_w:end_w]
        for key in data.get('gt_fields', []):
            data[key] = data[key][start_h:end_h, start_w:end_w]

        return data


class LimitLong:
    """
    Limit the long edge of image.

    If the long edge is larger than max_long, resize the long edge
    to max_long, while scale the short edge proportionally.

    If the long edge is smaller than min_long, resize the long edge
    to min_long, while scale the short edge proportionally.

    Args:
        max_long (int, optional): If the long edge of image is larger than max_long,
            it will be resize to max_long. Default: None.
        min_long (int, optional): If the long edge of image is smaller than min_long,
            it will be resize to min_long. Default: None.
    """

    def __init__(self, max_long=None, min_long=None):
        if max_long is not None:
            if not isinstance(max_long, int):
                raise TypeError(
                    "Type of `max_long` is invalid. It should be int, but it is {}"
                    .format(type(max_long)))
        if min_long is not None:
            if not isinstance(min_long, int):
                raise TypeError(
                    "Type of `min_long` is invalid. It should be int, but it is {}"
                    .format(type(min_long)))
        if (max_long is not None) and (min_long is not None):
            if min_long > max_long:
                raise ValueError(
                    '`max_long should not smaller than min_long, but they are {} and {}'
                    .format(max_long, min_long))
        self.max_long = max_long
        self.min_long = min_long

    def __call__(self, data):
        h, w = data['img'].shape[:2]
        long_edge = max(h, w)
        target = long_edge
        if (self.max_long is not None) and (long_edge > self.max_long):
            target = self.max_long
        elif (self.min_long is not None) and (long_edge < self.min_long):
            target = self.min_long

        if target != long_edge:
            data['trans_info'].append(('resize', data['img'].shape[0:2]))
            data['img'] = functional.resize_long(data['img'], target)
            for key in data.get('gt_fields', []):
                data[key] = functional.resize_long(data[key], self.long_size)

        return data


class RandomHorizontalFlip:
    """
    Flip an image horizontally with a certain probability.

    Args:
        prob (float, optional): A probability of horizontally flipping. Default: 0.5.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            data['img'] = functional.horizontal_flip(data['img'])
            for key in data.get('gt_fields', []):
                data[key] = functional.horizontal_flip(data[key])

        return data


class RandomBlur:
    """
    Blurring an image by a Gaussian function with a certain probability.

    Args:
        prob (float, optional): A probability of blurring an image. Default: 0.1.
    """

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, data):
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                data['img'] = cv2.GaussianBlur(data['img'], (radius, radius), 0,
                                               0)
                for key in data.get('gt_fields', []):
                    data[key] = cv2.GaussianBlur(data[key], (radius, radius), 0,
                                                 0)
        return data


if __name__ == "__main__":
    transforms = [RandomBlur(prob=1)]
    transforms = Compose(transforms)
    fg_path = '/ssd1/home/chenguowei01/github/PaddleSeg/contrib/matting/data/matting/human_matting/Distinctions-646/train/fg/13(2).png'
    alpha_path = fg_path.replace('fg', 'alpha')
    bg_path = '/ssd1/home/chenguowei01/github/PaddleSeg/contrib/matting/data/matting/human_matting/bg/unsplash_bg/attic/photo-1443884590026-2e4d21aee71c?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxMjA3fDB8MXxzZWFyY2h8Nzh8fGF0dGljfGVufDB8fHx8MTYyOTY4MDcxNQ&ixlib=rb-1.2.1&q=80&w=400.jpg'
    data = {}
    data['fg'] = cv2.imread(fg_path)
    data['bg'] = cv2.imread(bg_path)
    h, w, c = data['fg'].shape
    data['bg'] = cv2.resize(data['bg'], (w, h))
    alpha = cv2.imread(alpha_path)
    data['alpha'] = alpha[:, :, 0]
    alpha = alpha / 255.
    data['img'] = alpha * data['fg'] + (1 - alpha) * data['bg']

    data['gt_fields'] = ['fg', 'bg', 'alpha']
    data = transforms(data)
    cv2.imwrite('blur_alpha.png', data['alpha'])

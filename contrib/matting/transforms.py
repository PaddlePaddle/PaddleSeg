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
        data['trans_info'].append(('resize', data['img'].shape[-2:]))
        data['img'] = functional.resize(data['img'], self.target_size)
        for key in data.get('gt_fields', []):
            data[key] = functional.resize(data[key], self.target_size)
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
    Randomly crop with uncertain area as the center

    Args:
    crop_size (tuple|list): The size you want to crop from image.
    """

    def __init__(self, crop_size=((320, 320), (480, 480), (640, 640))):
        self.crop_size = crop_size

    def __call__(self, data):
        idex = np.random.randint(low=0, high=len(self.crop_size))
        crop_size = self.crop_size[idex]
        crop_center = np.where((data['alpha'] > 0) & (data['alpha'] < 255))
        center_h_array, center_w_array = crop_center
        delta_h = crop_size[1] // 2
        delta_w = crop_size[0] // 2

        if len(center_h_array) == 0:
            return data

        rand_ind = np.random.randint(len(center_h_array))
        center_h = center_h_array[rand_ind]
        center_w = center_w_array[rand_ind]

        start_h = max(0, center_h - delta_h)
        start_w = max(0, center_w - delta_w)
        end_h = min(data['img'].shape[0], start_h + crop_size[1])
        end_w = min(data['img'].shape[1], start_w + crop_size[0])

        data['img'] = data['img'][start_h:end_h, start_w:end_w]
        for key in data.get('gt_fields', []):
            data[key] = data[key][start_h:end_h, start_w:end_w]

        return data


if __name__ == "__main__":
    transforms = [LoadImages(), RandomCropByAlpha()]
    transforms = Compose(transforms)
    img_path = '/mnt/chenguowei01/github/PaddleSeg/data/matting/human_matting/train/image/0051115Q_000001_0062_001.png'
    bg_path = img_path.replace('image', 'bg')
    fg_path = img_path.replace('image', 'fg')
    alpha_path = img_path.replace('image', 'alpha')
    data = {}
    data['img'] = img_path
    data['fg'] = fg_path
    data['bg'] = bg_path
    data['alpha'] = alpha_path
    data['gt_fields'] = ['fg', 'bg', 'alpha']
    data = transforms(data)
    print(np.min(data['img']), np.max(data['img']))
    print(data['img'].shape, data['fg'].shape, data['bg'].shape,
          data['alpha'].shape)
    cv2.imwrite('crop_img.png', data['img'].transpose((1, 2, 0)))

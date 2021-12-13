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
from paddleseg.cvlibs import manager
from PIL import Image


@manager.TRANSFORMS.add_component
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
        if 'trans_info' not in data:
            data['trans_info'] = []
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


@manager.TRANSFORMS.add_component
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


@manager.TRANSFORMS.add_component
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
            if key == 'trimap':
                data[key] = functional.resize(data[key], self.target_size,
                                              cv2.INTER_NEAREST)
            else:
                data[key] = functional.resize(data[key], self.target_size)
        return data


@manager.TRANSFORMS.add_component
class RandomResize:
    """
    Resize image to a size determinned by `scale` and `size`.

    Args:
        size(tuple|list): The reference size to resize. A tuple or list with length 2.
        scale(tupel|list, optional): A range of scale base on `size`. A tuple or list with length 2. Default: None.
    """

    def __init__(self, size=None, scale=None):
        if isinstance(size, list) or isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(
                    '`size` should include 2 elements, but it is {}'.format(
                        size))
        elif size is not None:
            raise TypeError(
                "Type of `size` is invalid. It should be list or tuple, but it is {}"
                .format(type(size)))

        if scale is not None:
            if isinstance(scale, list) or isinstance(scale, tuple):
                if len(scale) != 2:
                    raise ValueError(
                        '`scale` should include 2 elements, but it is {}'.
                        format(scale))
            else:
                raise TypeError(
                    "Type of `scale` is invalid. It should be list or tuple, but it is {}"
                    .format(type(scale)))
        self.size = size
        self.scale = scale

    def __call__(self, data):
        h, w = data['img'].shape[:2]
        if self.scale is not None:
            scale = np.random.uniform(self.scale[0], self.scale[1])
        else:
            scale = 1.
        if self.size is not None:
            scale_factor = max(self.size[0] / w, self.size[1] / h)
        else:
            scale_factor = 1
        scale = scale * scale_factor

        w = int(round(w * scale))
        h = int(round(h * scale))
        data['img'] = functional.resize(data['img'], (w, h))
        for key in data.get('gt_fields', []):
            if key == 'trimap':
                data[key] = functional.resize(data[key], (w, h),
                                              cv2.INTER_NEAREST)
            else:
                data[key] = functional.resize(data[key], (w, h))
        return data


@manager.TRANSFORMS.add_component
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
            if key == 'trimap':
                data[key] = functional.resize_long(data[key], self.long_size,
                                                   cv2.INTER_NEAREST)
            else:
                data[key] = functional.resize_long(data[key], self.long_size)
        return data


@manager.TRANSFORMS.add_component
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
            if key == 'trimap':
                data[key] = functional.resize_short(data[key], self.short_size,
                                                    cv2.INTER_NEAREST)
            else:
                data[key] = functional.resize_short(data[key], self.short_size)
        return data


@manager.TRANSFORMS.add_component
class ResizeToIntMult:
    """
    Resize to some int muitple, d.g. 32.
    """

    def __init__(self, mult_int=32):
        self.mult_int = mult_int

    def __call__(self, data):
        data['trans_info'].append(('resize', data['img'].shape[0:2]))

        h, w = data['img'].shape[0:2]
        rw = w - w % self.mult_int
        rh = h - h % self.mult_int
        data['img'] = functional.resize(data['img'], (rw, rh))
        for key in data.get('gt_fields', []):
            if key == 'trimap':
                data[key] = functional.resize(data[key], (rw, rh),
                                              cv2.INTER_NEAREST)
            else:
                data[key] = functional.resize(data[key], (rw, rh))

        return data


@manager.TRANSFORMS.add_component
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


@manager.TRANSFORMS.add_component
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


@manager.TRANSFORMS.add_component
class RandomCrop:
    """
    Randomly crop

    Args:
    crop_size (tuple|list): The size you want to crop from image.
    """

    def __init__(self, crop_size=((320, 320), (480, 480), (640, 640))):
        if not isinstance(crop_size[0], (list, tuple)):
            crop_size = [crop_size]
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


@manager.TRANSFORMS.add_component
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

        data['trans_info'].append(('resize', data['img'].shape[0:2]))
        if target != long_edge:
            data['img'] = functional.resize_long(data['img'], target)
            for key in data.get('gt_fields', []):
                if key == 'trimap':
                    data[key] = functional.resize_long(data[key], target,
                                                       cv2.INTER_NEAREST)
                else:
                    data[key] = functional.resize_long(data[key], target)

        return data


@manager.TRANSFORMS.add_component
class LimitShort:
    """
    Limit the short edge of image.

    If the short edge is larger than max_short, resize the short edge
    to max_short, while scale the long edge proportionally.

    If the short edge is smaller than min_short, resize the short edge
    to min_short, while scale the long edge proportionally.

    Args:
        max_short (int, optional): If the short edge of image is larger than max_short,
            it will be resize to max_short. Default: None.
        min_short (int, optional): If the short edge of image is smaller than min_short,
            it will be resize to min_short. Default: None.
    """

    def __init__(self, max_short=None, min_short=None):
        if max_short is not None:
            if not isinstance(max_short, int):
                raise TypeError(
                    "Type of `max_short` is invalid. It should be int, but it is {}"
                    .format(type(max_short)))
        if min_short is not None:
            if not isinstance(min_short, int):
                raise TypeError(
                    "Type of `min_short` is invalid. It should be int, but it is {}"
                    .format(type(min_short)))
        if (max_short is not None) and (min_short is not None):
            if min_short > max_short:
                raise ValueError(
                    '`max_short should not smaller than min_short, but they are {} and {}'
                    .format(max_short, min_short))
        self.max_short = max_short
        self.min_short = min_short

    def __call__(self, data):
        h, w = data['img'].shape[:2]
        short_edge = min(h, w)
        target = short_edge
        if (self.max_short is not None) and (short_edge > self.max_short):
            target = self.max_short
        elif (self.min_short is not None) and (short_edge < self.min_short):
            target = self.min_short

        data['trans_info'].append(('resize', data['img'].shape[0:2]))
        if target != short_edge:
            data['img'] = functional.resize_short(data['img'], target)
            for key in data.get('gt_fields', []):
                if key == 'trimap':
                    data[key] = functional.resize_short(data[key], target,
                                                        cv2.INTER_NEAREST)
                else:
                    data[key] = functional.resize_short(data[key], target)

        return data


@manager.TRANSFORMS.add_component
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


@manager.TRANSFORMS.add_component
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
                    if key == 'trimap':
                        continue
                    data[key] = cv2.GaussianBlur(data[key], (radius, radius), 0,
                                                 0)
        return data


@manager.TRANSFORMS.add_component
class RandomDistort:
    """
    Distort an image with random configurations.

    Args:
        brightness_range (float, optional): A range of brightness. Default: 0.5.
        brightness_prob (float, optional): A probability of adjusting brightness. Default: 0.5.
        contrast_range (float, optional): A range of contrast. Default: 0.5.
        contrast_prob (float, optional): A probability of adjusting contrast. Default: 0.5.
        saturation_range (float, optional): A range of saturation. Default: 0.5.
        saturation_prob (float, optional): A probability of adjusting saturation. Default: 0.5.
        hue_range (int, optional): A range of hue. Default: 18.
        hue_prob (float, optional): A probability of adjusting hue. Default: 0.5.
    """

    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob

    def __call__(self, data):
        brightness_lower = 1 - self.brightness_range
        brightness_upper = 1 + self.brightness_range
        contrast_lower = 1 - self.contrast_range
        contrast_upper = 1 + self.contrast_range
        saturation_lower = 1 - self.saturation_range
        saturation_upper = 1 + self.saturation_range
        hue_lower = -self.hue_range
        hue_upper = self.hue_range
        ops = [
            functional.brightness, functional.contrast, functional.saturation,
            functional.hue
        ]
        random.shuffle(ops)
        params_dict = {
            'brightness': {
                'brightness_lower': brightness_lower,
                'brightness_upper': brightness_upper
            },
            'contrast': {
                'contrast_lower': contrast_lower,
                'contrast_upper': contrast_upper
            },
            'saturation': {
                'saturation_lower': saturation_lower,
                'saturation_upper': saturation_upper
            },
            'hue': {
                'hue_lower': hue_lower,
                'hue_upper': hue_upper
            }
        }
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob
        }

        im = data['img'].astype('uint8')
        im = Image.fromarray(im)
        for id in range(len(ops)):
            params = params_dict[ops[id].__name__]
            params['im'] = im
            prob = prob_dict[ops[id].__name__]
            if np.random.uniform(0, 1) < prob:
                im = ops[id](**params)
        data['img'] = np.asarray(im)

        for key in data.get('gt_fields', []):
            if key in ['alpha', 'trimap']:
                continue
            else:
                im = data[key].astype('uint8')
                im = Image.fromarray(im)
                for id in range(len(ops)):
                    params = params_dict[ops[id].__name__]
                    params['im'] = im
                    prob = prob_dict[ops[id].__name__]
                    if np.random.uniform(0, 1) < prob:
                        im = ops[id](**params)
                data[key] = np.asarray(im)
        return data


@manager.TRANSFORMS.add_component
class Padding:
    """
    Add bottom-right padding to a raw image or annotation image.

    Args:
        target_size (list|tuple): The target size after padding.
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.

    Raises:
        TypeError: When target_size is neither list nor tuple.
        ValueError: When the length of target_size is not 2.
    """

    def __init__(self, target_size, im_padding_value=(127.5, 127.5, 127.5)):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of target_size is invalid. It should be list or tuple, now is {}"
                .format(type(target_size)))

        self.target_size = target_size
        self.im_padding_value = im_padding_value

    def __call__(self, data):
        im_height, im_width = data['img'].shape[0], data['img'].shape[1]
        target_height = self.target_size[1]
        target_width = self.target_size[0]
        pad_height = max(0, target_height - im_height)
        pad_width = max(0, target_width - im_width)
        data['trans_info'].append(('padding', data['img'].shape[0:2]))
        if (pad_height == 0) and (pad_width == 0):
            return data
        else:
            data['img'] = cv2.copyMakeBorder(
                data['img'],
                0,
                pad_height,
                0,
                pad_width,
                cv2.BORDER_CONSTANT,
                value=self.im_padding_value)
            for key in data.get('gt_fields', []):
                if key in ['trimap', 'alpha']:
                    value = 0
                else:
                    value = self.im_padding_value
                data[key] = cv2.copyMakeBorder(
                    data[key],
                    0,
                    pad_height,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=value)
        return data

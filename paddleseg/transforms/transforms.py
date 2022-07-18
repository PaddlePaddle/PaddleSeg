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

import random
import math

import cv2
import numpy as np
from PIL import Image

from paddleseg.cvlibs import manager
from paddleseg.transforms import functional


@manager.TRANSFORMS.add_component
class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].

    Args:
        transforms (list): A list contains data pre-processing or augmentation. Empty list means only reading images, no transformation.
        to_rgb (bool, optional): If converting image to RGB color space. Default: True.

    Raises:
        TypeError: When 'transforms' is not a list.
        ValueError: when the length of 'transforms' is less than 1.
    """

    def __init__(self, transforms, to_rgb=True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.to_rgb = to_rgb

    def __call__(self, data):
        """
        Args:
            data: A dict to deal with. It may include keys: 'img', 'label', 'trans_info' and 'gt_fields'.
                'trans_info' reserve the image shape informating. And the 'gt_fields' save the key need to transforms
                together with 'img'

        Returns: A dict after process。
        """
        if 'img' not in data.keys():
            raise ValueError("`data` must include `img` key.")
        if isinstance(data['img'], str):
            data['img'] = cv2.imread(data['img']).astype('float32')
        if data['img'] is None:
            raise ValueError('Can\'t read The image file {}!'.format(data[
                'img']))
        if not isinstance(data['img'], np.ndarray):
            raise TypeError("Image type is not numpy.")
        if len(data['img'].shape) != 3:
            raise ValueError('Image is not 3-dimensional.')
        if 'label' in data.keys() and isinstance(data['label'], str):
            data['label'] = np.asarray(Image.open(data['label']))

        if self.to_rgb:
            data['img'] = cv2.cvtColor(data['img'], cv2.COLOR_BGR2RGB)

        # the `trans_info` will save the process of image shape, and will be used in evaluation and prediction.
        if 'trans_info' not in data.keys():
            data['trans_info'] = []

        for op in self.transforms:
            data = op(data)
        data['img'] = np.transpose(data['img'], (2, 0, 1))
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
class RandomVerticalFlip:
    """
    Flip an image vertically with a certain probability.

    Args:
        prob (float, optional): A probability of vertical flipping. Default: 0.1.
    """

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            data['img'] = functional.vertical_flip(data['img'])
            for key in data.get('gt_fields', []):
                data[key] = functional.vertical_flip(data[key])
        return data


@manager.TRANSFORMS.add_component
class Resize:
    """
    Resize an image.

    Args:
        target_size (list|tuple, optional): The target size of image. Default: (512, 512).
        interp (str, optional): The interpolation mode of resize is consistent with opencv.
            ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']. Note that when it is
            'RANDOM', a random interpolation mode would be specified. Default: "LINEAR".

    Raises:
        TypeError: When 'target_size' type is neither list nor tuple.
        ValueError: When "interp" is out of pre-defined methods ('NEAREST', 'LINEAR', 'CUBIC',
        'AREA', 'LANCZOS4', 'RANDOM').
    """

    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size=(512, 512), interp='LINEAR'):
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("`interp` should be one of {}".format(
                self.interp_dict.keys()))
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
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        data['img'] = functional.resize(data['img'], self.target_size,
                                        self.interp_dict[interp])
        for key in data.get('gt_fields', []):
            data[key] = functional.resize(data[key], self.target_size,
                                          cv2.INTER_NEAREST)

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
            data[key] = functional.resize_long(data[key], self.long_size,
                                               cv2.INTER_NEAREST)

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
            data[key] = functional.resize_short(data[key], self.short_size,
                                                cv2.INTER_NEAREST)

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
        data['trans_info'].append(('resize', data['img'].shape[0:2]))

        h, w = data['img'].shape[0], data['img'].shape[1]
        long_edge = max(h, w)
        target = long_edge
        if (self.max_long is not None) and (long_edge > self.max_long):
            target = self.max_long
        elif (self.min_long is not None) and (long_edge < self.min_long):
            target = self.min_long

        if target != long_edge:
            data['img'] = functional.resize_long(data['img'], target)
            for key in data.get('gt_fields', []):
                data[key] = functional.resize_long(data[key], target,
                                                   cv2.INTER_NEAREST)

        return data


@manager.TRANSFORMS.add_component
class ResizeRangeScaling:
    """
    Resize the long side of an image into a range, and then scale the other side proportionally.

    Args:
        min_value (int, optional): The minimum value of long side after resize. Default: 400.
        max_value (int, optional): The maximum value of long side after resize. Default: 600.
    """

    def __init__(self, min_value=400, max_value=600):
        if min_value > max_value:
            raise ValueError('min_value must be less than max_value, '
                             'but they are {} and {}.'.format(min_value,
                                                              max_value))
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, data):

        if self.min_value == self.max_value:
            random_size = self.max_value
        else:
            random_size = int(
                np.random.uniform(self.min_value, self.max_value) + 0.5)
        data['img'] = functional.resize_long(data['img'], random_size,
                                             cv2.INTER_LINEAR)
        for key in data.get('gt_fields', []):
            data[key] = functional.resize_long(data[key], random_size,
                                               cv2.INTER_NEAREST)

        return data


@manager.TRANSFORMS.add_component
class ResizeStepScaling:
    """
    Scale an image proportionally within a range.

    Args:
        min_scale_factor (float, optional): The minimum scale. Default: 0.75.
        max_scale_factor (float, optional): The maximum scale. Default: 1.25.
        scale_step_size (float, optional): The scale interval. Default: 0.25.

    Raises:
        ValueError: When min_scale_factor is smaller than max_scale_factor.
    """

    def __init__(self,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 scale_step_size=0.25):
        if min_scale_factor > max_scale_factor:
            raise ValueError(
                'min_scale_factor must be less than max_scale_factor, '
                'but they are {} and {}.'.format(min_scale_factor,
                                                 max_scale_factor))
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_step_size = scale_step_size

    def __call__(self, data):

        if self.min_scale_factor == self.max_scale_factor:
            scale_factor = self.min_scale_factor

        elif self.scale_step_size == 0:
            scale_factor = np.random.uniform(self.min_scale_factor,
                                             self.max_scale_factor)

        else:
            num_steps = int((self.max_scale_factor - self.min_scale_factor) /
                            self.scale_step_size + 1)
            scale_factors = np.linspace(self.min_scale_factor,
                                        self.max_scale_factor,
                                        num_steps).tolist()
            np.random.shuffle(scale_factors)
            scale_factor = scale_factors[0]
        w = int(round(scale_factor * data['img'].shape[1]))
        h = int(round(scale_factor * data['img'].shape[0]))

        data['img'] = functional.resize(data['img'], (w, h), cv2.INTER_LINEAR)
        for key in data.get('gt_fields', []):
            data[key] = functional.resize(data[key], (w, h), cv2.INTER_NEAREST)

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
        if not (isinstance(self.mean,
                           (list, tuple)) and isinstance(self.std,
                                                         (list, tuple))):
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

    def __init__(self,
                 target_size,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
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
        self.label_padding_value = label_padding_value

    def __call__(self, data):
        data['trans_info'].append(('padding', data['img'].shape[0:2]))
        im_height, im_width = data['img'].shape[0], data['img'].shape[1]
        if isinstance(self.target_size, int):
            target_height = self.target_size
            target_width = self.target_size
        else:
            target_height = self.target_size[1]
            target_width = self.target_size[0]
        pad_height = target_height - im_height
        pad_width = target_width - im_width
        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                'The size of image should be less than `target_size`, but the size of image ({}, {}) is larger than `target_size` ({}, {})'
                .format(im_width, im_height, target_width, target_height))
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
                data[key] = cv2.copyMakeBorder(
                    data[key],
                    0,
                    pad_height,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=self.label_padding_value)
        return data


@manager.TRANSFORMS.add_component
class PaddingByAspectRatio:
    """

    Args:
        aspect_ratio (int|float, optional): The aspect ratio = width / height. Default: 1.
    """

    def __init__(self,
                 aspect_ratio=1,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        self.aspect_ratio = aspect_ratio
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, data):

        img_height = data['img'].shape[0]
        img_width = data['img'].shape[1]
        ratio = img_width / img_height
        if ratio == self.aspect_ratio:
            return data
        elif ratio > self.aspect_ratio:
            img_height = int(img_width / self.aspect_ratio)
        else:
            img_width = int(img_height * self.aspect_ratio)
        padding = Padding(
            (img_width, img_height),
            im_padding_value=self.im_padding_value,
            label_padding_value=self.label_padding_value)
        return padding(data)


@manager.TRANSFORMS.add_component
class RandomPaddingCrop:
    """
    Crop a sub-image from a raw image and annotation image randomly. If the target cropping size
    is larger than original image, then the bottom-right padding will be added.

    Args:
        crop_size (tuple, optional): The target cropping size. Default: (512, 512).
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.

    Raises:
        TypeError: When crop_size is neither list nor tuple.
        ValueError: When the length of crop_size is not 2.
    """

    def __init__(self,
                 crop_size=(512, 512),
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        if isinstance(crop_size, list) or isinstance(crop_size, tuple):
            if len(crop_size) != 2:
                raise ValueError(
                    'Type of `crop_size` is list or tuple. It should include 2 elements, but it is {}'
                    .format(crop_size))
        else:
            raise TypeError(
                "The type of `crop_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(crop_size)))
        self.crop_size = crop_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, data):

        if isinstance(self.crop_size, int):
            crop_width = self.crop_size
            crop_height = self.crop_size
        else:
            crop_width = self.crop_size[0]
            crop_height = self.crop_size[1]

        img_height = data['img'].shape[0]
        img_width = data['img'].shape[1]

        if img_height == crop_height and img_width == crop_width:
            return data
        else:
            pad_height = max(crop_height - img_height, 0)
            pad_width = max(crop_width - img_width, 0)
            if (pad_height > 0 or pad_width > 0):
                data['img'] = cv2.copyMakeBorder(
                    data['img'],
                    0,
                    pad_height,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=self.im_padding_value)
                for key in data.get('gt_fields', []):
                    data[key] = cv2.copyMakeBorder(
                        data[key],
                        0,
                        pad_height,
                        0,
                        pad_width,
                        cv2.BORDER_CONSTANT,
                        value=self.label_padding_value)
                img_height = data['img'].shape[0]
                img_width = data['img'].shape[1]

            if crop_height > 0 and crop_width > 0:
                h_off = np.random.randint(img_height - crop_height + 1)
                w_off = np.random.randint(img_width - crop_width + 1)

                data['img'] = data['img'][h_off:(crop_height + h_off), w_off:(
                    w_off + crop_width), :]
                for key in data.get('gt_fields', []):
                    data[key] = data[key][h_off:(crop_height + h_off), w_off:(
                        w_off + crop_width)]
        return data


@manager.TRANSFORMS.add_component
class RandomCenterCrop:
    """
    Crops the given the input data at the center.
    Args:
        retain_ratio (tuple or list, optional): The length of the input list or tuple must be 2. Default: (0.5, 0.5).
        the first value is used for width and the second is for height.
        In addition, the minimum size of the cropped image is [width * retain_ratio[0], height * retain_ratio[1]].
    Raises:
        TypeError: When retain_ratio is neither list nor tuple. Default: None.
        ValueError: When the value of retain_ratio is not in [0-1].
    """

    def __init__(self, retain_ratio=(0.5, 0.5)):
        if isinstance(retain_ratio, list) or isinstance(retain_ratio, tuple):
            if len(retain_ratio) != 2:
                raise ValueError(
                    'When type of `retain_ratio` is list or tuple, it shoule include 2 elements, but it is {}'
                    .format(retain_ratio))
            if retain_ratio[0] > 1 or retain_ratio[1] > 1 or retain_ratio[
                    0] < 0 or retain_ratio[1] < 0:
                raise ValueError(
                    'Value of `retain_ratio` should be in [0, 1], but it is {}'.
                    format(retain_ratio))
        else:
            raise TypeError(
                "The type of `retain_ratio` is invalid. It should be list or tuple, but it is {}"
                .format(type(retain_ratio)))
        self.retain_ratio = retain_ratio

    def __call__(self, data):
        retain_width = self.retain_ratio[0]
        retain_height = self.retain_ratio[1]

        img_height = data['img'].shape[0]
        img_width = data['img'].shape[1]

        if retain_width == 1. and retain_height == 1.:
            return data
        else:
            randw = np.random.randint(img_width * (1 - retain_width))
            randh = np.random.randint(img_height * (1 - retain_height))
            offsetw = 0 if randw == 0 else np.random.randint(randw)
            offseth = 0 if randh == 0 else np.random.randint(randh)
            p0, p1, p2, p3 = offseth, img_height + offseth - randh, offsetw, img_width + offsetw - randw
            data['img'] = data['img'][p0:p1, p2:p3, :]
            for key in data.get('gt_fields', []):
                data[key] = data[key][p0:p1, p2:p3]

        return data


@manager.TRANSFORMS.add_component
class ScalePadding:
    """
        Add center padding to a raw image or annotation image,then scale the
        image to target size.

        Args:
            target_size (list|tuple, optional): The target size of image. Default: (512, 512).
            im_padding_value (list, optional): The padding value of raw image.
                Default: [127.5, 127.5, 127.5].
            label_padding_value (int, optional): The padding value of annotation image. Default: 255.

        Raises:
            TypeError: When target_size is neither list nor tuple.
            ValueError: When the length of target_size is not 2.
    """

    def __init__(self,
                 target_size=(512, 512),
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
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
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, data):
        height = data['img'].shape[0]
        width = data['img'].shape[1]

        new_im = np.zeros(
            (max(height, width), max(height, width), 3)) + self.im_padding_value
        if 'label' in data['gt_fields']:
            new_label = np.zeros((max(height, width), max(height, width)
                                  )) + self.label_padding_value

        if height > width:
            padding = int((height - width) / 2)
            new_im[:, padding:padding + width, :] = data['img']
            if 'label' in data['gt_fields']:
                new_label[:, padding:padding + width] = data['label']
        else:
            padding = int((width - height) / 2)
            new_im[padding:padding + height, :, :] = data['img']
            if 'label' in data['gt_fields']:
                new_label[padding:padding + height, :] = data['label']

        data['img'] = np.uint8(new_im)
        data['img'] = functional.resize(
            data['img'], self.target_size, interp=cv2.INTER_CUBIC)
        if 'label' in data['gt_fields']:
            data['label'] = np.uint8(new_label)
            data['label'] = functional.resize(
                data['label'], self.target_size, interp=cv2.INTER_CUBIC)
        return data


@manager.TRANSFORMS.add_component
class RandomNoise:
    """
    Superimposing noise on an image with a certain probability.

    Args:
        prob (float, optional): A probability of blurring an image. Default: 0.5.
        max_sigma(float, optional): The maximum value of standard deviation of the distribution.
            Default: 10.0.
    """

    def __init__(self, prob=0.5, max_sigma=10.0):
        self.prob = prob
        self.max_sigma = max_sigma

    def __call__(self, data):
        if random.random() < self.prob:
            mu = 0
            sigma = random.random() * self.max_sigma
            data['img'] = np.array(data['img'], dtype=np.float32)
            data['img'] += np.random.normal(mu, sigma, data['img'].shape)
            data['img'][data['img'] > 255] = 255
            data['img'][data['img'] < 0] = 0

        return data


@manager.TRANSFORMS.add_component
class RandomBlur:
    """
    Blurring an image by a Gaussian function with a certain probability.

    Args:
        prob (float, optional): A probability of blurring an image. Default: 0.1.
        blur_type(str, optional): A type of blurring an image,
            gaussian stands for cv2.GaussianBlur,
            median stands for cv2.medianBlur,
            blur stands for cv2.blur,
            random represents randomly selected from above.
            Default: gaussian.
    """

    def __init__(self, prob=0.1, blur_type="gaussian"):
        self.prob = prob
        self.blur_type = blur_type

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
                data['img'] = np.array(data['img'], dtype='uint8')
                if self.blur_type == "gaussian":
                    data['img'] = cv2.GaussianBlur(data['img'],
                                                   (radius, radius), 0, 0)
                elif self.blur_type == "median":
                    data['img'] = cv2.medianBlur(data['img'], radius)
                elif self.blur_type == "blur":
                    data['img'] = cv2.blur(data['img'], (radius, radius))
                elif self.blur_type == "random":
                    select = random.random()
                    if select < 0.3:
                        data['img'] = cv2.GaussianBlur(data['img'],
                                                       (radius, radius), 0)
                    elif select < 0.6:
                        data['img'] = cv2.medianBlur(data['img'], radius)
                    else:
                        data['img'] = cv2.blur(data['img'], (radius, radius))
                else:
                    data['img'] = cv2.GaussianBlur(data['img'],
                                                   (radius, radius), 0, 0)
        data['img'] = np.array(data['img'], dtype='float32')
        return data


@manager.TRANSFORMS.add_component
class RandomRotation:
    """
    Rotate an image randomly with padding.

    Args:
        max_rotation (float, optional): The maximum rotation degree. Default: 15.
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.
    """

    def __init__(self,
                 max_rotation=15,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        self.max_rotation = max_rotation
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, data):

        if self.max_rotation > 0:
            (h, w) = data['img'].shape[:2]
            do_rotation = np.random.uniform(-self.max_rotation,
                                            self.max_rotation)
            pc = (w // 2, h // 2)
            r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
            cos = np.abs(r[0, 0])
            sin = np.abs(r[0, 1])

            nw = int((h * sin) + (w * cos))
            nh = int((h * cos) + (w * sin))

            (cx, cy) = pc
            r[0, 2] += (nw / 2) - cx
            r[1, 2] += (nh / 2) - cy
            dsize = (nw, nh)
            data['img'] = cv2.warpAffine(
                data['img'],
                r,
                dsize=dsize,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.im_padding_value)
            for key in data.get('gt_fields', []):
                data[key] = cv2.warpAffine(
                    data[key],
                    r,
                    dsize=dsize,
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=self.label_padding_value)

        return data


@manager.TRANSFORMS.add_component
class RandomScaleAspect:
    """
    Crop a sub-image from an original image with a range of area ratio and aspect and
    then scale the sub-image back to the size of the original image.

    Args:
        min_scale (float, optional): The minimum area ratio of cropped image to the original image. Default: 0.5.
        aspect_ratio (float, optional): The minimum aspect ratio. Default: 0.33.
    """

    def __init__(self, min_scale=0.5, aspect_ratio=0.33):
        self.min_scale = min_scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, data):

        if self.min_scale != 0 and self.aspect_ratio != 0:
            img_height = data['img'].shape[0]
            img_width = data['img'].shape[1]
            for i in range(0, 10):
                area = img_height * img_width
                target_area = area * np.random.uniform(self.min_scale, 1.0)
                aspectRatio = np.random.uniform(self.aspect_ratio,
                                                1.0 / self.aspect_ratio)

                dw = int(np.sqrt(target_area * 1.0 * aspectRatio))
                dh = int(np.sqrt(target_area * 1.0 / aspectRatio))
                if (np.random.randint(10) < 5):
                    tmp = dw
                    dw = dh
                    dh = tmp

                if (dh < img_height and dw < img_width):
                    h1 = np.random.randint(0, img_height - dh)
                    w1 = np.random.randint(0, img_width - dw)

                    data['img'] = data['img'][h1:(h1 + dh), w1:(w1 + dw), :]
                    data['img'] = cv2.resize(
                        data['img'], (img_width, img_height),
                        interpolation=cv2.INTER_LINEAR)
                    for key in data.get('gt_fields', []):
                        data[key] = data[key][h1:(h1 + dh), w1:(w1 + dw)]
                        data[key] = cv2.resize(
                            data[key], (img_width, img_height),
                            interpolation=cv2.INTER_NEAREST)
                    break
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
        sharpness_range (float, optional): A range of sharpness. Default: 0.5.
        sharpness_prob (float, optional): A probability of adjusting saturation. Default: 0.
    """

    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5,
                 sharpness_range=0.5,
                 sharpness_prob=0):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob
        self.sharpness_range = sharpness_range
        self.sharpness_prob = sharpness_prob

    def __call__(self, data):

        brightness_lower = 1 - self.brightness_range
        brightness_upper = 1 + self.brightness_range
        contrast_lower = 1 - self.contrast_range
        contrast_upper = 1 + self.contrast_range
        saturation_lower = 1 - self.saturation_range
        saturation_upper = 1 + self.saturation_range
        hue_lower = -self.hue_range
        hue_upper = self.hue_range
        sharpness_lower = 1 - self.sharpness_range
        sharpness_upper = 1 + self.sharpness_range
        ops = [
            functional.brightness, functional.contrast, functional.saturation,
            functional.hue, functional.sharpness
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
            },
            'sharpness': {
                'sharpness_lower': sharpness_lower,
                'sharpness_upper': sharpness_upper,
            }
        }
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob,
            'sharpness': self.sharpness_prob
        }
        data['img'] = data['img'].astype('uint8')
        data['img'] = Image.fromarray(data['img'])
        for id in range(len(ops)):
            params = params_dict[ops[id].__name__]
            prob = prob_dict[ops[id].__name__]
            params['im'] = data['img']
            if np.random.uniform(0, 1) < prob:
                data['img'] = ops[id](**params)
        data['img'] = np.asarray(data['img']).astype('float32')
        return data


@manager.TRANSFORMS.add_component
class RandomAffine:
    """
    Affine transform an image with random configurations.

    Args:
        size (tuple, optional): The target size after affine transformation. Default: (224, 224).
        translation_offset (float, optional): The maximum translation offset. Default: 0.
        max_rotation (float, optional): The maximum rotation degree. Default: 15.
        min_scale_factor (float, optional): The minimum scale. Default: 0.75.
        max_scale_factor (float, optional): The maximum scale. Default: 1.25.
        im_padding_value (float, optional): The padding value of raw image. Default: (128, 128, 128).
        label_padding_value (int, optional): The padding value of annotation image. Default: (255, 255, 255).
    """

    def __init__(self,
                 size=(224, 224),
                 translation_offset=0,
                 max_rotation=15,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 im_padding_value=(128, 128, 128),
                 label_padding_value=255):
        self.size = size
        self.translation_offset = translation_offset
        self.max_rotation = max_rotation
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, data):

        w, h = self.size
        bbox = [0, 0, data['img'].shape[1] - 1, data['img'].shape[0] - 1]
        x_offset = (random.random() - 0.5) * 2 * self.translation_offset
        y_offset = (random.random() - 0.5) * 2 * self.translation_offset
        dx = (w - (bbox[2] + bbox[0])) / 2.0
        dy = (h - (bbox[3] + bbox[1])) / 2.0

        matrix_trans = np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])

        angle = random.random() * 2 * self.max_rotation - self.max_rotation
        scale = random.random() * (self.max_scale_factor - self.min_scale_factor
                                   ) + self.min_scale_factor
        scale *= np.mean(
            [float(w) / (bbox[2] - bbox[0]), float(h) / (bbox[3] - bbox[1])])
        alpha = scale * math.cos(angle / 180.0 * math.pi)
        beta = scale * math.sin(angle / 180.0 * math.pi)

        centerx = w / 2.0 + x_offset
        centery = h / 2.0 + y_offset
        matrix = np.array(
            [[alpha, beta, (1 - alpha) * centerx - beta * centery],
             [-beta, alpha, beta * centerx + (1 - alpha) * centery],
             [0, 0, 1.0]])

        matrix = matrix.dot(matrix_trans)[0:2, :]
        data['img'] = cv2.warpAffine(
            np.uint8(data['img']),
            matrix,
            tuple(self.size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.im_padding_value)
        for key in data.get('gt_fields', []):
            data[key] = cv2.warpAffine(
                np.uint8(data[key]),
                matrix,
                tuple(self.size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.label_padding_value)
        return data

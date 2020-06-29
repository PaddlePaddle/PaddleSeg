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

from .ops import *
import random
import numpy as np
import cv2
from collections import OrderedDict
from readers.reader import read_img


class Compose:
    """根据数据预处理/增强算子对输入数据进行操作。
       所有操作的输入图像流形状均是[H, W, C]，其中H为图像高，W为图像宽，C为图像通道数。

    Args:
        transforms (list): 数据预处理/增强算子。

    Raises:
        TypeError: transforms不是list对象
        ValueError: transforms元素个数小于1。

    """

    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        if len(transforms) < 1:
            raise ValueError('The length of transforms ' + \
                            'must be equal or larger than 1!')
        self.transforms = transforms

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (str/np.ndarray): 图像路径/图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息，dict中的字段如下：
                - shape_before_resize (tuple): 图像resize之前的大小（h, w）。
                - shape_before_padding (tuple): 图像padding之前的大小（h, w）。
            label (str/np.ndarray): 标注图像路径/标注图像np.ndarray数据。

        Returns:
            tuple: 根据网络所需字段所组成的tuple；字段由transforms中的最后一个数据预处理操作决定。
        """

        if im_info is None:
            im_info = dict()
        im = read_img(im)
        if im is None:
            raise ValueError('Can\'t read The image file {}!'.format(im))
        if label is not None:
            label = read_img(label)

        for op in self.transforms:
            outputs = op(im, im_info, label)
            im = outputs[0]
            if len(outputs) >= 2:
                im_info = outputs[1]
            if len(outputs) == 3:
                label = outputs[2]
        return outputs


class RandomHorizontalFlip:
    """以一定的概率对图像进行水平翻转。当存在标注图像时，则同步进行翻转。

    Args:
        prob (float): 随机水平翻转的概率。默认值为0.5。

    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息。
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if random.random() < self.prob:
            im = horizontal_flip(im)
            if label is not None:
                label = horizontal_flip(label)
        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class RandomVerticalFlip:
    """以一定的概率对图像进行垂直翻转。当存在标注图像时，则同步进行翻转。

    Args:
        prob (float): 随机垂直翻转的概率。默认值为0.1。
    """

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息。
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if random.random() < self.prob:
            im = vertical_flip(im)
            if label is not None:
                label = vertical_flip(label)
        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class Resize:
    """调整图像大小（resize），当存在标注图像时，则同步进行处理。

    - 当目标大小（target_size）类型为int时，根据插值方式，
      将图像resize为[target_size, target_size]。
    - 当目标大小（target_size）类型为list或tuple时，根据插值方式，
      将图像resize为target_size, target_size的输入应为[w, h]或（w, h）。

    Args:
        target_size (int/list/tuple): 目标大小
        interp (str): resize的插值方式，与opencv的插值方式对应，
            可选的值为['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4']，默认为"LINEAR"。

    Raises:
        TypeError: target_size不是int/list/tuple。
        ValueError:  target_size为list/tuple时元素个数不等于2。
        AssertionError: interp的取值不在['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4']之内
    """

    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size, interp='LINEAR'):
        self.interp = interp
        assert interp in self.interp_dict, "interp should be one of {}".format(
            self.interp_dict.keys())
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    'when target is list or tuple, it should include 2 elements, but it is {}'
                    .format(target_size))
        elif not isinstance(target_size, int):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or tuple, now is {}"
                .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息。
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
                其中，im_info跟新字段为：
                    -shape_before_resize (tuple): 保存resize之前图像的形状(h, w）。

        Raises:
            ZeroDivisionError: im的短边为0。
            TypeError: im不是np.ndarray数据。
            ValueError: im不是3维nd.ndarray。
        """
        if im_info is None:
            im_info = OrderedDict()
        im_info['shape_before_resize'] = im.shape[:2]

        if not isinstance(im, np.ndarray):
            raise TypeError("ResizeImage: image type is not np.ndarray.")
        if len(im.shape) != 3:
            raise ValueError('ResizeImage: image is not 3-dimensional.')
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if float(im_size_min) == 0:
            raise ZeroDivisionError('ResizeImage: min size of image is 0')

        if isinstance(self.target_size, int):
            resize_w = self.target_size
            resize_h = self.target_size
        else:
            resize_w = self.target_size[0]
            resize_h = self.target_size[1]
        im_scale_x = float(resize_w) / float(im_shape[1])
        im_scale_y = float(resize_h) / float(im_shape[0])

        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp_dict[self.interp])
        if label is not None:
            label = cv2.resize(
                label,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp_dict['NEAREST'])
        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class ResizeByLong:
    """对图像长边resize到固定值，短边按比例进行缩放。当存在标注图像时，则同步进行处理。

    Args:
        long_size (int): resize后图像的长边大小。
    """

    def __init__(self, long_size):
        self.long_size = long_size

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息。
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
                其中，im_info新增字段为：
                    -shape_before_resize (tuple): 保存resize之前图像的形状(h, w）。
        """
        if im_info is None:
            im_info = OrderedDict()

        im_info['shape_before_resize'] = im.shape[:2]
        im = resize_long(im, self.long_size)
        if label is not None:
            label = resize_long(label, self.long_size, cv2.INTER_NEAREST)

        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class ResizeRangeScaling:
    """对图像长边随机resize到指定范围内，短边按比例进行缩放。当存在标注图像时，则同步进行处理。

    Args:
        min_value (int): 图像长边resize后的最小值。默认值400。
        max_value (int): 图像长边resize后的最大值。默认值600。

    Raises:
        ValueError: min_value大于max_value
    """

    def __init__(self, min_value=400, max_value=600):
        if min_value > max_value:
            raise ValueError('min_value must be less than max_value, '
                             'but they are {} and {}.'.format(
                                 min_value, max_value))
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息。
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if self.min_value == self.max_value:
            random_size = self.max_value
        else:
            random_size = int(
                np.random.uniform(self.min_value, self.max_value) + 0.5)
        value = max(im.shape[0], im.shape[1])
        scale = float(random_size) / float(value)
        im = cv2.resize(
            im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(
                label, (0, 0),
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST)

        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class ResizeStepScaling:
    """对图像按照某一个比例resize，这个比例以scale_step_size为步长
    在[min_scale_factor, max_scale_factor]随机变动。当存在标注图像时，则同步进行处理。

    Args:
        min_scale_factor（float), resize最小尺度。默认值0.75。
        max_scale_factor (float), resize最大尺度。默认值1.25。
        scale_step_size (float), resize尺度范围间隔。默认值0.25。

    Raises:
        ValueError: min_scale_factor大于max_scale_factor
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

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息。
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
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

        im = cv2.resize(
            im, (0, 0),
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(
                label, (0, 0),
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_NEAREST)

        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class Clip:
    """
    对图像上超出一定范围的数据进行裁剪。

    Args:
        min_val (list): 裁剪的下限，小于min_val的数值均设为min_val. 默认值[0, 0, 0].
        max_val (list): 裁剪的上限，大于max_val的数值均设为max_val. 默认值[255.0, 255.0, 255.0]
    """

    def __init__(self, min_val=[0, 0, 0], max_val=[255.0, 255.0, 255.0]):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, im, im_info=None, label=None):
        if isinstance(self.min_val, list) and isinstance(self.max_val, list):
            for k in range(im.shape[2]):
                np.clip(
                    im[:, :, k],
                    self.min_val[k],
                    self.max_val[k],
                    out=im[:, :, k])
        else:
            raise TypeError('min_val and max_val must be list')

        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class Normalize:
    """对图像进行标准化。
    1.图像像素归一化到区间 [0.0, 1.0]。
    2.对图像进行减均值除以标准差操作。

    Args:
        min_val (list): 图像数据集的最小值。默认值[0, 0, 0].
        max_val (list): 图像数据集的最大值。默认值[255.0, 255.0, 255.0]
        mean (list): 图像数据集的均值。默认值[0.5, 0.5, 0.5].
        std (list): 图像数据集的标准差。默认值[0.5, 0.5, 0.5].

    Raises:
        ValueError: mean或std不是list对象。std包含0。
    """

    def __init__(self,
                 min_val=[0, 0, 0],
                 max_val=[255.0, 255.0, 255.0],
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5]):
        self.min_val = min_val
        self.max_val = max_val
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, list) and isinstance(self.std, list)):
            raise ValueError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息。
            label (np.ndarray): 标注图像np.ndarray数据。

         Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """

        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]

        im = normalize(im, self.min_val, self.max_val, mean, std)

        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class Padding:
    """对图像或标注图像进行padding，padding方向为右和下。
    根据提供的值对图像或标注图像进行padding操作。

    Args:
        target_size (int/list/tuple): padding后图像的大小。
        im_padding_value (list): 图像padding的值。默认为127.5。
        label_padding_value (int): 标注图像padding的值。默认值为255。

    Raises:
        TypeError: target_size不是int/list/tuple。
        ValueError:  target_size为list/tuple时元素个数不等于2。
    """

    def __init__(self,
                 target_size,
                 im_padding_value=127.5,
                 label_padding_value=255):
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    'when target is list or tuple, it should include 2 elements, but it is {}'
                    .format(target_size))
        elif not isinstance(target_size, int):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or tuple, now is {}"
                .format(type(target_size)))
        self.target_size = target_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息。
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
                其中，im_info新增字段为：
                    -shape_before_padding (tuple): 保存padding之前图像的形状(h, w）。

        Raises:
            ValueError: 输入图像im或label的形状大于目标值
        """
        if im_info is None:
            im_info = OrderedDict()
        im_info['shape_before_padding'] = im.shape[:2]

        im_height, im_width = im.shape[0], im.shape[1]
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
                'the size of image should be less than target_size, but the size of image ({}, {}), is larger than target_size ({}, {})'
                .format(im_width, im_height, target_width, target_height))
        else:
            im = np.pad(
                im,
                pad_width=((0, pad_height), (0, pad_width), (0, 0)),
                mode='constant',
                constant_values=(self.im_padding_value, self.im_padding_value))
            if label is not None:
                label = np.pad(
                    label,
                    pad_width=((0, pad_height), (0, pad_width)),
                    mode='constant',
                    constant_values=(self.label_padding_value,
                                     self.label_padding_value))
        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class RandomPaddingCrop:
    """对图像和标注图进行随机裁剪，当所需要的裁剪尺寸大于原图时，则进行padding操作。

    Args:
        crop_size（int or list or tuple): 裁剪图像大小。默认为512。
        im_padding_value (list): 图像padding的值。默认为127.5
        label_padding_value (int): 标注图像padding的值。默认值为255。

    Raises:
        TypeError: crop_size不是int/list/tuple。
        ValueError:  target_size为list/tuple时元素个数不等于2。
    """

    def __init__(self,
                 crop_size=512,
                 im_padding_value=127.5,
                 label_padding_value=255):
        if isinstance(crop_size, list) or isinstance(crop_size, tuple):
            if len(crop_size) != 2:
                raise ValueError(
                    'when crop_size is list or tuple, it should include 2 elements, but it is {}'
                    .format(crop_size))
        elif not isinstance(crop_size, int):
            raise TypeError(
                "Type of crop_size is invalid. Must be Integer or List or tuple, now is {}"
                .format(type(crop_size)))
        self.crop_size = crop_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息。
            label (np.ndarray): 标注图像np.ndarray数据。

         Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if isinstance(self.crop_size, int):
            crop_width = self.crop_size
            crop_height = self.crop_size
        else:
            crop_width = self.crop_size[0]
            crop_height = self.crop_size[1]

        img_height = im.shape[0]
        img_width = im.shape[1]

        if img_height == crop_height and img_width == crop_width:
            if label is None:
                return (im, im_info)
            else:
                return (im, im_info, label)
        else:
            pad_height = max(crop_height - img_height, 0)
            pad_width = max(crop_width - img_width, 0)
            if (pad_height > 0 or pad_width > 0):
                im = np.pad(
                    im,
                    pad_width=((0, pad_height), (0, pad_width), (0, 0)),
                    mode='constant',
                    constant_values=(self.im_padding_value,
                                     self.im_padding_value))
                if label is not None:
                    label = np.pad(
                        label,
                        pad_width=((0, pad_height), (0, pad_width)),
                        mode='constant',
                        constant_values=(self.label_padding_value,
                                         self.label_padding_value))
                img_height = im.shape[0]
                img_width = im.shape[1]

            if crop_height > 0 and crop_width > 0:
                h_off = np.random.randint(img_height - crop_height + 1)
                w_off = np.random.randint(img_width - crop_width + 1)

                im = im[h_off:(crop_height + h_off), w_off:(
                    w_off + crop_width), :]
                if label is not None:
                    label = label[h_off:(crop_height + h_off), w_off:(
                        w_off + crop_width)]
        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class RandomBlur:
    """以一定的概率对图像进行高斯模糊。

    Args：
        prob (float): 图像模糊概率。默认为0.1。
    """

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息。
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
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
                im = cv2.GaussianBlur(im, (radius, radius), 0, 0)

        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class RandomScaleAspect:
    """裁剪并resize回原始尺寸的图像和标注图像。
    按照一定的面积比和宽高比对图像进行裁剪，并reszie回原始图像的图像，当存在标注图时，同步进行。

    Args：
        min_scale (float)：裁取图像占原始图像的面积比，0-1，默认0返回原图。默认为0.5。
        aspect_ratio (float): 裁取图像的宽高比范围，非负，默认0返回原图。默认为0.33。
    """

    def __init__(self, min_scale=0.5, aspect_ratio=0.33):
        self.min_scale = min_scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, im, im_info=None, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息。
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if self.min_scale != 0 and self.aspect_ratio != 0:
            img_height = im.shape[0]
            img_width = im.shape[1]
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

                    im = im[h1:(h1 + dh), w1:(w1 + dw), :]
                    label = label[h1:(h1 + dh), w1:(w1 + dw)]
                    im = cv2.resize(
                        im, (img_width, img_height),
                        interpolation=cv2.INTER_LINEAR)
                    label = cv2.resize(
                        label, (img_width, img_height),
                        interpolation=cv2.INTER_NEAREST)
                    break
        if label is None:
            return (im, im_info)
        else:
            return (im, im_info, label)


class ArrangeSegmenter:
    """获取训练/验证/预测所需的信息。

    Args:
        mode (str): 指定数据用于何种用途，取值范围为['train', 'eval', 'test', 'quant']。

    Raises:
        ValueError: mode的取值不在['train', 'eval', 'test', 'quant']之内
    """

    def __init__(self, mode):
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode should be defined as one of ['train', 'eval', 'test', 'quant']!"
            )
        self.mode = mode

    def __call__(self, im, im_info, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息。
            label (np.ndarray): 标注图像np.ndarray数据。

        Returns:
            tuple: 当mode为'train'或'eval'时，返回的tuple为(im, label)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当mode为'test'时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；当mode为
                'quant'时，返回的tuple为(im,)，为图像np.ndarray数据。
        """
        im = permute(im, False)
        if self.mode == 'train' or self.mode == 'eval':
            label = label[np.newaxis, :, :]
            return (im, label)
        elif self.mode == 'test':
            return (im, im_info)
        else:
            return (im, )

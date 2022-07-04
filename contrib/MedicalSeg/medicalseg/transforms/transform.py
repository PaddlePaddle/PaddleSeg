# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# Todo: Add transform components here

import math
import random
import numpy as np
import numbers
import collections

from medicalseg.cvlibs import manager
from medicalseg.transforms import functional as F


@manager.TRANSFORMS.add_component
class Compose:
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].

    Args:
        transforms (list): A list contains data pre-processing or augmentation. Empty list means only reading images, no transformation.

    Raises:
        TypeError: When 'transforms' is not a list.
        ValueError: when the length of 'transforms' is less than 1.
    """

    def __init__(self, transforms, isnhwd=True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.isnhwd = isnhwd

    def __call__(self, im, label=None, isnhwd=True):
        """
        Args:
            im (str|np.ndarray): It is either image path or image object.
            label (str|np.ndarray): It is either label path or label ndarray.
            isnhwd:data format。

        Returns:
            (tuple). A tuple including image, image info, and label after transformation.
        """
        if isinstance(im, str):
            im = np.load(im)
        if isinstance(label, str):
            label = np.load(label)
        if im is None:
            raise ValueError('Can\'t read The image file {}!'.format(im))

        for op in self.transforms:
            outputs = op(im, label)
            im = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]
        if self.isnhwd:
            im = np.expand_dims(im, axis=0)

        if im.max() > 0:
            im = im / im.max()

        return (im, label)


@manager.TRANSFORMS.add_component
class Resize3D:
    """Resize the input numpy ndarray to the given size.
    Args:
        size
        order (int, optional): Desired order
    """

    def __init__(self, size, order=1):
        """
        resize
        """
        if isinstance(size, int):
            self.size = size
        elif isinstance(size, collections.abc.Iterable) and len(size) == 3:
            if type(size) == list:
                size = tuple(size)
            self.size = size
        else:
            raise ValueError('Unknown inputs for size: {}'.format(size))
        self.order = order
        super().__init__()

    def __call__(self, img, label=None):
        """
        Args:
            img (numpy ndarray): Image to be scaled.
            label (numpy ndarray) : Label to be scaled
        Returns:
            numpy ndarray: Rescaled image.
            numpy ndarray: Rescaled label.
        """
        img = F.resize_3d(img, self.size, self.order)
        if label is not None:
            label = F.resize_3d(label, self.size, 0)
        return img, label


@manager.TRANSFORMS.add_component
class RandomRotation3D:
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, rotate_planes=[[0, 1], [0, 2], [1, 2]]):
        """
        init
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.rotate_planes = rotate_planes

        super().__init__()

    def get_params(self, degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])
        r_plane = self.rotate_planes[random.randint(
            0, len(self.rotate_planes) - 1)]

        return angle, r_plane

    def __call__(self, img, label=None):
        """
        Args:
            img (numpy ndarray): 3D Image to be flipped.
            label (numpy ndarray): 3D Label to be flipped.
        Returns:
            (np.array). Image after transformation.
        """
        angle, r_plane = self.get_params(self.degrees)
        img = F.rotate_3d(img, r_plane, angle)
        if label is not None:
            label = F.rotate_3d(label, r_plane, angle)
        return img, label


@manager.TRANSFORMS.add_component
class RandomFlip3D:
    """Flip an 3D image with a certain probability.
    Args:
        prob (float, optional): A probability of vertical flipping. Default: 0.1.
    """

    def __init__(self, prob=0.5, flip_axis=[0, 1, 2]):
        """
        init
        """
        self.prob = prob
        self.flip_axis = flip_axis

        super().__init__()

    def __call__(self, img, label=None):
        """
        Args:
            img (numpy ndarray): 3D Image to be flipped.
            label (numpy ndarray): 3D Label to be flipped.
        Returns:
            (np.array). Image after transformation.
        """
        if isinstance(self.flip_axis, (tuple, list)):
            flip_axis = self.flip_axis[random.randint(0,
                                                      len(self.flip_axis) - 1)]
        else:
            flip_axis = self.flip_axis

        if random.random() < self.prob:
            img = F.flip_3d(img, axis=flip_axis)
            if label is not None:
                label = F.flip_3d(label, axis=flip_axis)
        return img, label


@manager.TRANSFORMS.add_component
class RandomResizedCrop3D:
    """
    先Crop再Resize至预设尺寸
    scale: 切出cube的体积与原图体积的比值范围
    ratio: 切出cube的每一边长的抖动范围
    size:  resize的目标尺寸
    interpolation: [1-5]， skimage.zoom的order数。注意分割模式下label的order统一为0
    pre_crop: bool，如果为True，则先切一个目标尺寸左右的cube，再resize，通常用于滑窗模式；
                    如果为False，则从原图上扣一个与原图接近的cube，再resize至目标尺寸
    nonzero_mask，如果为True，则只在label mask有效(非0)区域内进行滑窗
                  如果为False，则在image整个区域内进行滑窗
    """

    def __init__(self, size, scale=(0.8, 1.2), ratio=(3. / 4., 4. / 3.), \
        interpolation=1, pre_crop=False, nonzero_mask=False):
        """
        init
        """
        if isinstance(size, (tuple, list)):
            assert len(size) == 3, \
                "Size must contain THREE number when it is a tuple or list, got {}.".format(len(size))
            self.size = size
        elif isinstance(size, int):
            self.size = (size, size, size)
        else:
            print("Size must be a list or tuple, got {}.".format(type(size)))

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.pre_crop = pre_crop
        self.nonzero_mask = nonzero_mask

        super().__init__()

    def get_params(self, img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (numpy ndarray): Image to be cropped. d, h, w
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        params_ret = collections.namedtuple('params_ret',
                                            ['i', 'j', 'k', 'd', 'h', 'w'])
        for attempt in range(10):
            volume = img.shape[0] * img.shape[1] * img.shape[2]
            target_volume = random.uniform(*scale) * volume
            aspect_ratio = random.uniform(*ratio)

            d = int(round((target_volume * aspect_ratio)**(1 / 3)))
            h = int(round((target_volume / aspect_ratio)**(1 / 3)))
            w = img.shape[2]

            if random.random() < 0.5:
                d, h, w = random.sample([d, h, w], k=3)

            if w <= img.shape[2] and h <= img.shape[1] and d <= img.shape[0]:
                i = random.randint(0, img.shape[0] - d)
                j = random.randint(0, img.shape[1] - h)
                k = random.randint(0, img.shape[2] - w)
                return params_ret(i, j, k, d, h, w)

        # Fallback
        w = min(img.shape[0], img.shape[1], img.shape[2])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        k = (img.shape[2] - w) // 2
        return params_ret(i, j, k, w, w, w)

    def pre_crop_util(self, img, label=None):
        """
        pre crop逻辑
        """
        if self.pre_crop:
            crop_size = (np.random.uniform(low=self.scale[0], high=self.scale[1], size=3) * \
                                 self.size).round().astype("int")

            if self.nonzero_mask:
                mask_voxel_coords = np.where(label != 0)
                minzidx = int(np.min(mask_voxel_coords[0]))
                maxzidx = int(np.max(mask_voxel_coords[0])) + 1
                minyidx = int(np.min(mask_voxel_coords[1]))
                maxyidx = int(np.max(mask_voxel_coords[1])) + 1
                minxidx = int(np.min(mask_voxel_coords[2]))
                maxxidx = int(np.max(mask_voxel_coords[2])) + 1

                masked_shape = np.array(
                    [maxzidx - minzidx, maxyidx - minyidx, maxxidx - minxidx])
                crop_z, crop_y, crop_x = np.minimum(masked_shape, crop_size)
                z_start = np.random.randint(masked_shape[0] - crop_z +
                                            1) + minzidx
                y_start = np.random.randint(masked_shape[1] - crop_y +
                                            1) + minyidx
                x_start = np.random.randint(masked_shape[2] - crop_x +
                                            1) + minxidx

                z_end = z_start + crop_z
                y_end = y_start + crop_y
                x_end = x_start + crop_x
            else:
                crop_z, crop_y, crop_x = np.minimum(img.shape[:3], crop_size)
                z_start = np.random.randint(img.shape[0] - crop_z + 1)
                y_start = np.random.randint(img.shape[1] - crop_y + 1)
                x_start = np.random.randint(img.shape[2] - crop_x + 1)

                z_end = z_start + crop_z
                y_end = y_start + crop_y
                x_end = x_start + crop_x

            img = img[z_start:z_end, y_start:y_end, x_start:x_end]
            if label is not None:
                label = label[z_start:z_end, y_start:y_end, x_start:x_end]

        return img, label

    def __call__(self, img, label=None):
        """
        Args:
            img (numpy ndarray): Image to be cropped and resized.
        Returns:
            numpy ndarray: Randomly cropped and resized image.
        """
        img, label = self.pre_crop_util(img, label)
        i, j, k, d, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop_3d(img, i, j, k, d, h, w, self.size,
                                self.interpolation)
        if label is not None:
            label = F.resized_crop_3d(label, i, j, k, d, h, w, self.size, 0)

        return img, label


@manager.TRANSFORMS.add_component
class BinaryMaskToConnectComponent:
    """Got the connect compoent from binary mask
    Args:
        minimum_volume (int, default=0): The minimum volume of the connected component to be retained
    """

    def __init__(self, minimum_volume=0):
        """
        resize
        """
        self.minimum_volume = minimum_volume
        super().__init__()

    def __call__(self, pred, label=None):
        """
        Args:
            img (numpy ndarray): Image to be scaled.
            label (numpy ndarray) : Label to be scaled
        Returns:
            numpy ndarray: Rescaled image.
            numpy ndarray: Rescaled label.
        """
        pred = F.extract_connect_compoent(pred, self.minimum_volume)
        if label is not None:
            label = F.extract_connect_compoent(label, self.minimum_volume)
        return pred, label


@manager.TRANSFORMS.add_component
class TopkLargestConnectComponent:
    """Keep topk largest connect component sorted by volume nums, remove others.
    Args:
        k (int, default=1): k
    """

    def __init__(self, k=1):
        """
        resize
        """
        self.k = k
        super().__init__()

    def __call__(self, pred, label=None):
        """
        Args:
            img (numpy ndarray): Image to be scaled.
            label (numpy ndarray) : Label to be scaled
        Returns:
            numpy ndarray: Rescaled image.
            numpy ndarray: Rescaled label.
        """
        pred = F.extract_connect_compoent(pred)
        pred[pred > self.k] = 0
        return pred, label


@manager.TRANSFORMS.add_component
class RandomRotation4D:
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, rotate_planes=[[0, 1], [0, 2], [1, 2]]):
        """
        init
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.rotate_planes = rotate_planes

        super().__init__()

    def get_params(self, degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])
        r_plane = self.rotate_planes[random.randint(
            0, len(self.rotate_planes) - 1)]

        return angle, r_plane

    def __call__(self, img, label=None):
        """
        Args:
            img (numpy ndarray): 3D Image to be flipped.
            label (numpy ndarray): 3D Label to be flipped.
        Returns:
            (np.array). Image after transformation.
        """
        angle, r_plane = self.get_params(self.degrees)

        img = F.rotate_4d(img, r_plane, angle)
        if label is not None:
            label = F.rotate_4d(label, map(lambda s: s - 1, r_plane), angle)
        return img, label


@manager.TRANSFORMS.add_component
class RandomFlip4D:
    """Flip an 4D image with a certain probability.
    Args:
        prob (float, optional): A probability of vertical flipping. Default: 0.1.
    """

    def __init__(self, prob=0.5, flip_axis=[0, 1, 2]):
        """
        init
        """
        self.prob = prob
        self.flip_axis = flip_axis

        super().__init__()

    def __call__(self, img, label=None):
        """
        Args:
            img (numpy ndarray): 4D Image to be flipped.
            label (numpy ndarray): 4D Label to be flipped.
        Returns:
            (np.array). Image after transformation.
        """
        if isinstance(self.flip_axis, (tuple, list)):
            flip_axis = self.flip_axis[random.randint(0,
                                                      len(self.flip_axis) - 1)]
        else:
            flip_axis = self.flip_axis

        if random.random() < self.prob:
            img = F.flip_3d(img, axis=flip_axis)
            if label is not None:
                label = F.flip_3d(label, axis=flip_axis - 1)
        return img, label


@manager.TRANSFORMS.add_component
class RandomCrop4D:
    """
    RandomCrop至预设尺寸
    scale: 切出cube的体积与原图体积的比值范围
    ratio: 切出cube的每一边长的抖动范围
    size:  resize的目标尺寸
    interpolation: [1-5]， skimage.zoom的order数。注意分割模式下label的order统一为0
    pre_crop: bool，如果为True，则先切一个目标尺寸左右的cube，再resize，通常用于滑窗模式；
                    如果为False，则从原图上扣一个与原图接近的cube，再resize至目标尺寸
    nonzero_mask，如果为True，则只在label mask有效(非0)区域内进行滑窗
                  如果为False，则在image整个区域内进行滑窗
    """

    def __init__(self,
                 size,
                 scale=(0.8, 1.2),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=1,
                 pre_crop=False,
                 nonzero_mask=False):
        """
        init
        """
        if isinstance(size, (tuple, list)):
            assert len(size) == 3, \
                "Size must contain THREE number when it is a tuple or list, got {}.".format(len(size))
            self.size = size
        elif isinstance(size, int):
            self.size = (size, size, size)
        else:
            print("Size must be a list or tuple, got {}.".format(type(size)))

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.size = size
        self.pre_crop = pre_crop
        self.nonzero_mask = nonzero_mask

        super().__init__()

    def get_params(self, img, scale, ratio, size):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (numpy ndarray): Image to be cropped. d, h, w
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        params_ret = collections.namedtuple('params_ret',
                                            ['i', 'j', 'k', 'd', 'h', 'w'])

        d = size
        h = size
        w = size
        i = random.randint(0, img.shape[1] - d)
        j = random.randint(0, img.shape[2] - h)
        k = random.randint(0, img.shape[3] - w)
        return params_ret(i, j, k, d, h, w)

    def __call__(self, img, label=None):
        """
        Args:
            img (numpy ndarray): Image to be cropped and resized.
        Returns:
            numpy ndarray: Randomly cropped and resized image.
        """

        i, j, k, d, h, w = self.get_params(img, self.scale, self.ratio,
                                           self.size)

        img = F.crop_4d(img, i, j, k, d, h, w)
        if label is not None:
            label = F.crop_3d(label, i, j, k, d, h, w)

        return img, label

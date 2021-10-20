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
class Compose_DA:
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

    def __call__(self, im, label=None):
        """
        Args:
            im (str|np.ndarray): It is either image path or image object.
            label (str|np.ndarray): It is either label path or label ndarray.

        Returns:
            (tuple). A tuple including image, image info, and label after transformation.
        """
        if isinstance(im, str):
            im = cv2.imread(im).astype('float32')
        if isinstance(label, str):
            label = np.asarray(Image.open(label))
        if im is None:
            raise ValueError('Can\'t read The image file {}!'.format(im))
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        logger.add()
        
        for op in self.transforms:
            outputs = op(im, label)
            im = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]
        im = im[:, :, ::-1]  # change to BGR
        # print('before subtract IMG_MEAN', np.mean(im, axis=[0,1]))
        IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434),
                            dtype=np.float32)
        im = im - IMG_MEAN
        # print('im mean after subtract IMG_MEAN', np.mean(im, axis=[0,1]))
        im = np.transpose(im, (2, 0, 1))
        return (im, label)

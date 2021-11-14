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
import random

from paddleseg.cvlibs import manager


@manager.TRANSFORMS.add_component
class CutHeight:
    def __init__(self, cut_height=0):
        self.cut_height = cut_height

    def __call__(self, im, label=None):
        im = im[self.cut_height:, :, :]
        if label is None:
            return (im, )
        else:
            label = label[self.cut_height:, :]
            return (im, label)


@manager.TRANSFORMS.add_component
class LaneRandomRotation:
    def __init__(self,
                 max_rotation=10,
                 im_padding_value=(0, 0, 0),
                 label_padding_value=0):
        self.max_rotation = max_rotation
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, label=None):
        v = random.random()
        if v < 0.5:
            (h, w) = im.shape[:2]
            do_rotation = np.random.uniform(-self.max_rotation,
                                            self.max_rotation)
            pc = (w // 2, h // 2)
            r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
            dsize = (w, h)
            im = cv2.warpAffine(
                im,
                r,
                dsize=dsize,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.im_padding_value)
            if label is not None:
                label = cv2.warpAffine(
                    label,
                    r,
                    dsize=dsize,
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=self.label_padding_value)

        if label is None:
            return (im, )
        else:
            return (im, label)

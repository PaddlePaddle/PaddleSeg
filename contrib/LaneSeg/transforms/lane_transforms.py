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

from paddleseg.cvlibs import manager


@manager.TRANSFORMS.add_component
class LaneRandomRotation:
    """
    Rotate an image randomly with padding.

    Args:
        max_rotation (float, optional): The maximum rotation degree. Default: 15.
        keeping_size (bool, optional): Whether or not to holding image size. Default: False.
        im_padding_value (list, optional): The padding value of raw image.
            Default: [127.5, 127.5, 127.5].
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.
    """

    def __init__(self,
                 max_rotation=15,
                 keeping_size=False,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        self.max_rotation = max_rotation
        self.keeping_size = keeping_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): The Image data.
            label (np.ndarray, optional): The label data. Default: None.

        Returns:
            (tuple). When label is None, it returns (im, ), otherwise it returns (im, label).
        """

        if self.max_rotation > 0:
            (h, w) = im.shape[:2]
            do_rotation = np.random.uniform(-self.max_rotation,
                                            self.max_rotation)
            pc = (w // 2, h // 2)
            r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
            if self.keeping_size:
                dsize = (w, h)
            else:
                cos = np.abs(r[0, 0])
                sin = np.abs(r[0, 1])

                nw = int((h * sin) + (w * cos))
                nh = int((h * cos) + (w * sin))
                (cx, cy) = pc
                r[0, 2] += (nw / 2) - cx
                r[1, 2] += (nh / 2) - cy
                dsize = (nw, nh)
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


@manager.TRANSFORMS.add_component
class SubImgCrop:
    """
    crop an image from four forwards.

    Args:
        offset_top (int, optional): The cut height for image from up to down. Default: 0.
        offset_bottom (int, optional): The cut height for image from down to up . Default: 0.
        offset_left (int, optional): The cut height for image from left to right. Default: 0.
        offset_right (int, optional): The cut width for image from right to left. Default: 0.
    """

    def __init__(self,
                 offset_top=0,
                 offset_bottom=0,
                 offset_left=0,
                 offset_right=0):
        self.offset_top = offset_top
        self.offset_bottom = offset_bottom
        self.offset_left = offset_left
        self.offset_right = offset_right

    def __call__(self, im, label=None):
        if self.offset_top < 0 or self.offset_bottom < 0 or self.offset_left < 0 or self.offset_right < 0:
            raise Exception(
                "offset_top, offset_bottom, offset_left,  offset_right must equal or greater zero"
            )

        if self.offset_top > 0 and self.offset_top < im.shape[0]:
            im = im[self.offset_top:, :, :]
            if label is not None:
                label = label[self.offset_top:, :]

        if self.offset_bottom > 0 and self.offset_bottom < im.shape[0]:
            im = im[:-self.offset_bottom, :, :]
            if label is not None:
                label = label[:-self.offset_bottom, :]

        if self.offset_left > 0 and self.offset_left < im.shape[1]:
            im = im[:, self.offset_left:, :]
            if label is not None:
                label = label[:, self.left_w_off:]

        if self.offset_right > 0 and self.offset_right < im.shape[1]:
            im = im[:, :-self.offset_right, :]
            if label is not None:
                label = label[:, :-self.offset_right]

        if label is None:
            return (im, )
        else:
            return (im, label)

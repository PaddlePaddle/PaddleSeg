# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# This implementation refers to: https://github.com/facebookresearch/segment-anything

import numpy as np
import PIL
from PIL import Image
from copy import deepcopy
from typing import Tuple

import paddle
from paddle.nn import functional as F
from paddle.vision.transforms.functional import resize  # type: ignore


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not (isinstance(pic, paddle.Tensor) or isinstance(pic, np.ndarray)):
        raise TypeError(f"pic should be Tensor or ndarray. Got {type(pic)}.")

    elif isinstance(pic, paddle.Tensor):
        if pic.ndim not in {2, 3}:
            raise ValueError(
                f"pic should be 2/3 dimensional. Got {pic.ndim} dimensions.")

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (CHW)
            pic = pic.unsqueeze(0)

        # check number of channels
        if pic.shape[-3] > 4:
            raise ValueError(
                f"pic should not have > 4 channels. Got {pic.shape[-3]} channels."
            )

    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError(
                f"pic should be 2/3 dimensional. Got {pic.ndim} dimensions.")

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)

        # check number of channels
        if pic.shape[-1] > 4:
            raise ValueError(
                f"pic should not have > 4 channels. Got {pic.shape[-1]} channels."
            )

    npimg = pic
    if isinstance(pic, paddle.Tensor):
        if pic.is_floating_point() and mode != "F":
            pic = (pic * 255).astype("uint8")
        npimg = np.transpose(pic.cpu().numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError(
            "Input pic must be a paddle.Tensor or NumPy ndarray, not {type(npimg)}"
        )

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = "L"
        elif npimg.dtype == np.int16:
            expected_mode = "I;16"
        elif npimg.dtype == np.int32:
            expected_mode = "I"
        elif npimg.dtype == np.float32:
            expected_mode = "F"
        if mode is not None and mode != expected_mode:
            raise ValueError(
                f"Incorrect mode ({mode}) supplied "
                f"for input type {np.dtype}. Should be {expected_mode}")
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ["LA"]
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError(
                f"Only modes {permitted_2_channel_modes} are supported for 2D inputs"
            )

        if mode is None and npimg.dtype == np.uint8:
            mode = "LA"

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ["RGBA", "CMYK", "RGBX"]
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(
                f"Only modes {permitted_4_channel_modes} are supported for 4D inputs"
            )

        if mode is None and npimg.dtype == np.uint8:
            mode = "RGBA"
    else:
        permitted_3_channel_modes = ["RGB", "YCbCr", "HSV"]
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(
                f"Only modes {permitted_3_channel_modes} are supported for 3D inputs"
            )
        if mode is None and npimg.dtype == np.uint8:
            mode = "RGB"

    if mode is None:
        raise TypeError(f"Input type {npimg.dtype} is not supported")

    return Image.fromarray(npimg, mode=mode)


class ResizeLongestSide:
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched paddle tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1],
                                                self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray,
                     original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray,
                    original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape([-1, 2, 2]), original_size)
        return boxes.reshape([-1, 4])

    def apply_image_paddle(self,
                           image: paddle.Tensor) -> paddle.Tensor:  # not used
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1],
                                                self.target_length)
        return F.interpolate(
            image,
            target_size,
            mode="bilinear",
            align_corners=False,  # todo: missing this flag antialias=True
        )

    def apply_coords_paddle(self,
                            coords: paddle.Tensor,
                            original_size: Tuple[int, ...]) -> paddle.Tensor:
        """
        Expects a paddle tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).to(paddle.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_paddle(self,
                           boxes: paddle.Tensor,
                           original_size: Tuple[int, ...]) -> paddle.Tensor:
        """
        Expects a paddle tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_paddle(
            boxes.reshape([-1, 2, 2]), original_size)
        return boxes.reshape([-1, 4])

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int,
                             long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

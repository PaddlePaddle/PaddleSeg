# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import cv2

from eiseg import logger


def has_sitk():
    try:
        import SimpleITK

        return True
    except ImportError:
        return False


if has_sitk():
    import SimpleITK as sitk


def dcm_reader(path):
    logger.debug(f"opening medical image {path}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([path])
    image = reader.Execute()
    img = sitk.GetArrayFromImage(image)
    logger.debug(f"scan shape is {img.shape}")
    if len(img.shape) == 4:
        img = img[0]
    # WHC
    img = np.transpose(img, [1, 2, 0])
    return img.astype(np.int32)


def windowlize(scan, ww, wc):
    wl = wc - ww / 2
    wh = wc + ww / 2
    res = scan.copy()
    res = res.astype(np.float32)
    res = np.clip(res, wl, wh)
    res = (res - wl) / ww * 255
    res = res.astype(np.uint8)
    # print("++", res.shape)
    # for idx in range(res.shape[-1]):
    # TODO: 支持3d或者改调用
    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

    return res


# def open_nii(niiimg_path):
#     if IPT_SITK == True:
#         sitk_image = sitk.ReadImage(niiimg_path)
#         return _nii2arr(sitk_image)
#     else:
#         raise ImportError("can't import SimpleITK!")


#
# def _nii2arr(sitk_image):
#     if IPT_SITK == True:
#         img = sitk.GetArrayFromImage(sitk_image).transpose((1, 2, 0))
#         return img
#     else:
#         raise ImportError("can't import SimpleITK!")
#
#
# def slice_img(img, index):
#     if index == 0:
#         return sample_norm(
#             cv2.merge(
#                 [
#                     np.uint16(img[:, :, index]),
#                     np.uint16(img[:, :, index]),
#                     np.uint16(img[:, :, index + 1]),
#                 ]
#             )
#         )
#     elif index == img.shape[2] - 1:
#         return sample_norm(
#             cv2.merge(
#                 [
#                     np.uint16(img[:, :, index - 1]),
#                     np.uint16(img[:, :, index]),
#                     np.uint16(img[:, :, index]),
#                 ]
#             )
#         )
#     else:
#         return sample_norm(
#             cv2.merge(
#                 [
#                     np.uint16(img[:, :, index - 1]),
#                     np.uint16(img[:, :, index]),
#                     np.uint16(img[:, :, index + 1]),
#                 ]
#             )
#         )

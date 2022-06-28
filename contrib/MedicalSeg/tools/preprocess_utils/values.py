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

# TODO add clip [0.9%, 99.1%]
import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
import tools.preprocess_utils.global_var as global_var

gpu_tag = global_var.get_value('USE_GPU')
if gpu_tag:
    import cupy as np
    if int(np.__version__.split(".")[0]) < 10:
        if global_var.get_value("ALERTED_HUNORM_NUMPY") is not True:
            print(
                f"[Warning] Running HUNorm preprocess with cupy requires cupy version >= 10.0.0 . Installed version is {np.__version__}. Using numpy for HUNorm. Other preprocess operations are still run on GPU."
            )
            global_var.set_value("ALERTED_HUNORM_NUMPY", True)
        import numpy as np
else:
    import numpy as np


def label_remap(label, map_dict=None):
    """
    Convert labels using label map

    label: 3D numpy/cupy array in [z, y, x] order.
    map_dict: the label transfer map dict. key is the original label, value is the remaped one.
    """

    if not isinstance(label, np.ndarray):
        image = np.array(label)

    for key, val in map_dict.items():
        label[label == key] = val

    return label


def normalize(image, min_val=None, max_val=None):
    "Normalize the image with given min_val and max val "
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    if min_val is None and max_val is None:
        image = (image - image.min()) / (image.max() - image.min())
    else:
        image = (image - min_val) / (max_val - min_val)
    np.clip(image, 0, 1, out=image)

    return image


def HUnorm(image, HU_min=-1200, HU_max=600, HU_nan=-2000):
    """
    Convert CT HU unit into uint8 values. First bound HU values by predfined min
    and max, and then normalize. Due to paddle.nn.conv3D doesn't support uint8, we need to convert
    the returned image as float32.

    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    HU_min: float, min HU value.
    HU_max: float, max HU value.
    HU_nan: float, value for nan in the raw CT image.
    """

    if not isinstance(image, np.ndarray): 
        image = np.array(image)
    image = np.nan_to_num(image, copy=False, nan=HU_nan)

    # normalize to [0, 1]
    image = (image - HU_min) / ((HU_max - HU_min) / 255)
    np.clip(image, 0, 255, out=image)

    return image

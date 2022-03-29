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

# Reference: https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/utils/save_annotation.py

import os

import cv2
import numpy as np
from PIL import Image as PILImage

# Refence: https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/colormap.py#L14
_COLORS = np.array([
    0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184,
    0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078, 0.184, 0.300,
    0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000, 1.000, 0.500, 0.000,
    0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000,
    1.000, 0.333, 0.333, 0.000, 0.333, 0.667, 0.000, 0.333, 1.000, 0.000, 0.667,
    0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000, 1.000, 0.333, 0.000,
    1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667,
    0.500, 0.000, 1.000, 0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333,
    0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500,
    0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
    0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000, 0.000,
    0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333, 0.333, 1.000,
    0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333,
    1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000,
    0.333, 1.000, 1.000, 0.667, 1.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000,
    0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167,
    0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
    0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333,
    0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000,
    1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.857, 0.857, 0.857, 1.000,
    1.000, 1.000
]).astype(np.float32).reshape(-1, 3)


def random_color(rgb=False, maximum=255):
    """
    Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/colormap.py#L111

    Args:
        rgb (bool, optional): whether to return RGB colors or BGR colors. Default: False.
        maximum (int, optional): either 255 or 1. Default: 255.

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def cityscape_colormap():
    """Get CityScapes colormap"""
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    colormap = colormap[:, ::-1]
    return colormap


def visualize_semantic(semantic, save_path, colormap, image=None, weight=0.5):
    """
    Save semantic segmentation results.

    Args:
        semantic(np.ndarray): The result semantic segmenation results, shape is (h, w).
        save_path(str): The save path.
        colormap(np.ndarray): A color map for visualization.
        image(np.ndarray, optional): Origin image to prediction, merge semantic with
            image if provided. Default: None.
        weight(float, optional): The image weight when merge semantic with image. Default: 0.5.
    """
    semantic = semantic.astype('uint8')
    colored_semantic = colormap[semantic]
    if image is not None:
        colored_semantic = cv2.addWeighted(image, weight, colored_semantic,
                                           1 - weight, 0)
    cv2.imwrite(save_path, colored_semantic)


def visualize_instance(instance, save_path, stuff_id=0, image=None, weight=0.5):
    """
    Save instance segmentation results.

    Args:
        instance(np.ndarray): The instance segmentation results, shape is (h, w).
        save_path(str): The save path.
        stuff_id(int, optional): Id for background that not want to plot.
        image(np.ndarray, optional): Origin image to prediction, merge instance with
            image if provided. Default: None.
        weight(float, optional): The image weight when merge instance with image. Default: 0.5.
    """
    # Add color map for instance segmentation result.
    ids = np.unique(instance)
    num_colors = len(ids)
    colormap = np.zeros((num_colors, 3), dtype=np.uint8)
    # Maps label to continuous value
    for i in range(num_colors):
        instance[instance == ids[i]] = i
        colormap[i, :] = random_color(maximum=255)
        if ids[i] == stuff_id:
            colormap[i, :] = np.array([0, 0, 0])
    colored_instance = colormap[instance]

    if image is not None:
        colored_instance = cv2.addWeighted(image, weight, colored_instance,
                                           1 - weight, 0)
    cv2.imwrite(save_path, colored_instance)


def visualize_panoptic(panoptic,
                       save_path,
                       label_divisor,
                       colormap,
                       image=None,
                       weight=0.5,
                       ignore_index=255):
    """
    Save panoptic segmentation results.

    Args:
        panoptic(np.ndarray): The panoptic segmentation results, shape is (h, w).
        save_path(str): The save path.
        label_divisor(int): Used to convert panoptic id = semantic id * label_divisor + instance_id.
        colormap(np.ndarray): A color map for visualization.
        image(np.ndarray, optional): Origin image to prediction, merge panoptic with
            image if provided. Default: None.
        weight(float, optional): The image weight when merge panoptic with image. Default: 0.5.
        ignore_index(int, optional): Specifies a target value that is ignored. Default: 255.
    """
    colored_panoptic = np.zeros(
        (panoptic.shape[0], panoptic.shape[1], 3), dtype=np.uint8)
    taken_colors = set((0, 0, 0))

    def _random_color(base, max_dist=30):
        color = base + np.random.randint(
            low=-max_dist, high=max_dist + 1, size=3)
        return tuple(np.maximum(0, np.minimum(255, color)))

    for lab in np.unique(panoptic):
        mask = panoptic == lab

        ignore_mask = panoptic == ignore_index
        ins_mask = panoptic > label_divisor
        if lab > label_divisor:
            base_color = colormap[lab // label_divisor]
        elif lab != ignore_index:
            base_color = colormap[lab]
        else:
            continue
        if tuple(base_color) not in taken_colors:
            taken_colors.add(tuple(base_color))
            color = base_color
        else:
            while True:
                color = _random_color(base_color)
                if color not in taken_colors:
                    taken_colors.add(color)
                    break
        colored_panoptic[mask] = color

    if image is not None:
        colored_panoptic = cv2.addWeighted(image, weight, colored_panoptic,
                                           1 - weight, 0)
    cv2.imwrite(save_path, colored_panoptic)

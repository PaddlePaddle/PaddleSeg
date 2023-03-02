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

import cv2
import numpy as np

from .encode import decode_pan_id

# Refence: https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/colormap.py#L14
_COLORS = np.array([
    0.000, 0.447, 0.741,
    0.850, 0.325, 0.098,
    0.929, 0.694, 0.125,
    0.494, 0.184, 0.556,
    0.466, 0.674, 0.188,
    0.301, 0.745, 0.933,
    0.635, 0.078, 0.184,
    0.300, 0.300, 0.300,
    0.600, 0.600, 0.600,
    1.000, 0.000, 0.000,
    1.000, 0.500, 0.000,
    0.749, 0.749, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 1.000,
    0.667, 0.000, 1.000,
    0.333, 0.333, 0.000,
    0.333, 0.667, 0.000,
    0.333, 1.000, 0.000,
    0.667, 0.333, 0.000,
    0.667, 0.667, 0.000,
    0.667, 1.000, 0.000,
    1.000, 0.333, 0.000,
    1.000, 0.667, 0.000,
    1.000, 1.000, 0.000,
    0.000, 0.333, 0.500,
    0.000, 0.667, 0.500,
    0.000, 1.000, 0.500,
    0.333, 0.000, 0.500,
    0.333, 0.333, 0.500,
    0.333, 0.667, 0.500,
    0.333, 1.000, 0.500,
    0.667, 0.000, 0.500,
    0.667, 0.333, 0.500,
    0.667, 0.667, 0.500,
    0.667, 1.000, 0.500,
    1.000, 0.000, 0.500,
    1.000, 0.333, 0.500,
    1.000, 0.667, 0.500,
    1.000, 1.000, 0.500,
    0.000, 0.333, 1.000,
    0.000, 0.667, 1.000,
    0.000, 1.000, 1.000,
    0.333, 0.000, 1.000,
    0.333, 0.333, 1.000,
    0.333, 0.667, 1.000,
    0.333, 1.000, 1.000,
    0.667, 0.000, 1.000,
    0.667, 0.333, 1.000,
    0.667, 0.667, 1.000,
    0.667, 1.000, 1.000,
    1.000, 0.000, 1.000,
    1.000, 0.333, 1.000,
    1.000, 0.667, 1.000,
    0.333, 0.000, 0.000,
    0.500, 0.000, 0.000,
    0.667, 0.000, 0.000,
    0.833, 0.000, 0.000,
    1.000, 0.000, 0.000,
    0.000, 0.167, 0.000,
    0.000, 0.333, 0.000,
    0.000, 0.500, 0.000,
    0.000, 0.667, 0.000,
    0.000, 0.833, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 0.167,
    0.000, 0.000, 0.333,
    0.000, 0.000, 0.500,
    0.000, 0.000, 0.667,
    0.000, 0.000, 0.833,
    0.000, 0.000, 1.000,
    0.000, 0.000, 0.000,
    0.143, 0.143, 0.143,
    0.857, 0.857, 0.857,
    1.000, 1.000, 1.000
]).astype(np.float32).reshape(-1, 3)    # yapf: disable


def random_color(rgb=True, maximum=255):
    """
    Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/colormap.py#L111

    Args:
        rgb (bool, optional): Whether or not to return RGB colors or BGR colors. Default: True.
        maximum (int, optional): Either 255 or 1. Default: 255.

    Returns:
        ndarray: a vector of 3 numbers.
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def visualize_semantic(semantic, colormap=None, image=None, weight=0.5):
    """
    Visualize semantic segmentation results.

    Args:
        semantic (np.ndarray): The semantic segmenation results, whose shape is (h, w).
        colormap (np.ndarray, optional): The color map used for visualization. Default: None.
        image (np.ndarray, optional): The original input image. Merge the segmentation result 
            with `image` if it is provided. Default: None.
        weight (float, optional): The weight of original image when performing image merge. 
            Default: 0.5.
    """
    semantic = semantic.astype('uint8')
    if colormap is None:
        colormap = _COLORS * 255
    colored_semantic = colormap[semantic]
    if image is not None:
        colored_semantic = cv2.addWeighted(image, weight, colored_semantic,
                                           1 - weight, 0)
    return colored_semantic


def visualize_instance(instance,
                       ignore_ins_id=0,
                       image=None,
                       weight=0.5,
                       use_random_color=False):
    """
    Visualize instance segmentation results.

    Args:
        instance (np.ndarray): The instance segmentation results, whose shape is (h, w).
        ignore_ins_id (int, optional): ID for background area that you do not want to plot.
        image (np.ndarray, optional): The original input image. Merge the segmentation result 
            with `image` if it is provided. Default: None.
        weight (float, optional): The weight of original image when performing image merge. 
            Default: 0.5.
        use_random_color (bool, optional): Whether or not to use randomly selected color to
            visualize instances. Default: False.
    """
    # Add color map for instance segmentation result
    ids = np.unique(instance)
    num_colors = len(ids)
    colormap = np.zeros((num_colors, 3), dtype=np.uint8)
    # Make a copy to avoid in-place modification on the input
    instance = instance.copy()
    # Maps label to continuous value
    for i in range(num_colors):
        instance[instance == ids[i]] = i
        if use_random_color:
            colormap[i, :] = random_color(maximum=255)
        else:
            colormap[i, :] = _COLORS[i] * 255
        if ids[i] == ignore_ins_id:
            colormap[i, :] = np.array([0, 0, 0])
    colored_instance = colormap[instance]

    if image is not None:
        colored_instance = cv2.addWeighted(image, weight, colored_instance,
                                           1 - weight, 0)
    return colored_instance


def visualize_panoptic(panoptic,
                       label_divisor,
                       colormap=None,
                       image=None,
                       weight=0.5,
                       ignore_index=255):
    """
    Visualize panoptic segmentation results.

    Args:
        panoptic (np.ndarray): The panoptic segmentation results, whose shape is (h, w).
        label_divisor (int): Used for conversion between semantic IDs and panoptic IDs.
        colormap (np.ndarray, optional): The color map used for visualization. Default: None.
        image (np.ndarray, optional): The original input image. Merge the segmentation result 
            with `image` if it is provided. Default: None.
        weight (float, optional): The weight of original image when performing image merge. 
            Default: 0.5.
        ignore_index (int, optional): The class ID to be ignored. Default: 255.
    """

    def _random_color(base, max_dist=50):
        # TODO: Jitter in HSV color space
        color = base.astype('int32') + np.random.randint(
            low=-max_dist, high=max_dist + 1, size=3)
        # Return an immutable object
        return tuple(np.maximum(0, np.minimum(255, color)))

    colored_panoptic = np.zeros(
        (panoptic.shape[0], panoptic.shape[1], 3), dtype=np.uint8)
    if colormap is None:
        colormap = _COLORS * 255
    taken_colors = set((0, 0, 0))

    for lab in np.unique(panoptic):
        mask = panoptic == lab
        # XXX: The conversion strategy is hard-coded
        cat_id, ins_id = decode_pan_id(lab, label_divisor)
        if cat_id != ignore_index:
            base_color = colormap[cat_id]
        else:
            continue
        color = base_color
        if tuple(base_color) not in taken_colors:
            taken_colors.add(tuple(base_color))
        if ins_id > 0:
            while True:
                # FIXME: This might result in an infinite loop
                color = _random_color(base_color)
                if color not in taken_colors:
                    taken_colors.add(color)
                    break
        colored_panoptic[mask] = color

    if image is not None:
        colored_panoptic = cv2.addWeighted(image, weight, colored_panoptic,
                                           1 - weight, 0)
    return colored_panoptic

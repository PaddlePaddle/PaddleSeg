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

import numpy as np


class PanopticTargetGenerator(object):
    """
    Generates panoptic training target for Panoptic-DeepLab.
    Annotation is assumed to have Cityscapes format.

    Args:
        ignore_index (int): The ignore label for semantic segmentation.
        rgb2id (Function): Function, panoptic label is encoded in a colored image, this function convert color to the
            corresponding panoptic label.
        thing_list (list): A list of thing classes
        sigma (int, optional): The sigma for Gaussian kernel. Default: 8.
        ignore_stuff_in_offset (bool, optional): Whether to ignore stuff region when training the offset branch. Default: False.
        small_instance_area (int, optional): Indicates largest area for small instances. Default: 0.
        small_instance_weight (int, optional): Indicates semantic loss weights for small instances. Default: 1.
        ignore_crowd_in_semantic (bool, optional): Whether to ignore crowd region in semantic segmentation branch,
            crowd region is ignored in the original TensorFlow implementation. Default: False.
    """

    def __init__(self,
                 ignore_index,
                 rgb2id,
                 thing_list,
                 sigma=8,
                 ignore_stuff_in_offset=False,
                 small_instance_area=0,
                 small_instance_weight=1,
                 ignore_crowd_in_semantic=False):
        self.ignore_index = ignore_index
        self.rgb2id = rgb2id
        self.thing_list = thing_list
        self.ignore_stuff_in_offset = ignore_stuff_in_offset
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight
        self.ignore_crowd_in_semantic = ignore_crowd_in_semantic

        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def __call__(self, panoptic, segments):
        """Generates the training target.
        reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py
        reference: https://github.com/facebookresearch/detectron2/blob/master/datasets/prepare_panoptic_fpn.py#L18

        Args:
            panoptic (np.ndarray): Colored image encoding panoptic label.
            segments (list): A list of dictionary containing information of every segment, it has fields:
                - id: panoptic id, after decoding `panoptic`.
                - category_id: semantic class id.
                - area: segment area.
                - bbox: segment bounding box.
                - iscrowd: crowd region.

        Returns:
            A dictionary with fields:
                - semantic: Tensor, semantic label, shape=(H, W).
                - foreground: Tensor, foreground mask label, shape=(H, W).
                - center: Tensor, center heatmap, shape=(1, H, W).
                - center_points: List, center coordinates, with tuple (y-coord, x-coord).
                - offset: Tensor, offset, shape=(2, H, W), first dim is (offset_y, offset_x).
                - semantic_weights: Tensor, loss weight for semantic prediction, shape=(H, W).
                - center_weights: Tensor, ignore region of center prediction, shape=(H, W), used as weights for center
                    regression 0 is ignore, 1 is has instance. Multiply this mask to loss.
                - offset_weights: Tensor, ignore region of offset prediction, shape=(H, W), used as weights for offset
                    regression 0 is ignore, 1 is has instance. Multiply this mask to loss.
        """
        panoptic = self.rgb2id(panoptic)
        height, width = panoptic.shape[0], panoptic.shape[1]
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_index
        foreground = np.zeros_like(panoptic, dtype=np.uint8)
        center = np.zeros((1, height, width), dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord = np.ones_like(panoptic, dtype=np.float32)
        x_coord = np.ones_like(panoptic, dtype=np.float32)
        y_coord = np.cumsum(y_coord, axis=0) - 1
        x_coord = np.cumsum(x_coord, axis=1) - 1
        # Generate pixel-wise loss weights
        semantic_weights = np.ones_like(panoptic, dtype=np.uint8)
        # 0: ignore, 1: has instance
        # three conditions for a region to be ignored for instance branches:
        # (1) It is labeled as `ignore_index`
        # (2) It is crowd region (iscrowd=1)
        # (3) (Optional) It is stuff region (for offset branch)
        center_weights = np.zeros_like(panoptic, dtype=np.uint8)
        offset_weights = np.zeros_like(panoptic, dtype=np.uint8)
        for seg in segments:
            cat_id = seg["category_id"]
            if self.ignore_crowd_in_semantic:
                if not seg['iscrowd']:
                    semantic[panoptic == seg["id"]] = cat_id
            else:
                semantic[panoptic == seg["id"]] = cat_id
            if cat_id in self.thing_list:
                foreground[panoptic == seg["id"]] = 1
            if not seg['iscrowd']:
                # Ignored regions are not in `segments`.
                # Handle crowd region.
                center_weights[panoptic == seg["id"]] = 1
                if self.ignore_stuff_in_offset:
                    # Handle stuff region.
                    if cat_id in self.thing_list:
                        offset_weights[panoptic == seg["id"]] = 1
                else:
                    offset_weights[panoptic == seg["id"]] = 1
            if cat_id in self.thing_list:
                # find instance center
                mask_index = np.where(panoptic == seg["id"])
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_weights[panoptic ==
                                     seg["id"]] = self.small_instance_weight

                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[
                    1])
                center_pts.append([center_y, center_x])

                # generate center heatmap
                y, x = int(center_y), int(center_x)
                # outside image boundary
                if x < 0 or y < 0 or \
                        x >= width or y >= height:
                    continue
                sigma = self.sigma
                # upper left
                ul = int(np.round(x - 3 * sigma - 1)), int(
                    np.round(y - 3 * sigma - 1))
                # bottom right
                br = int(np.round(x + 3 * sigma + 2)), int(
                    np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], width)
                aa, bb = max(0, ul[1]), min(br[1], height)
                center[0, aa:bb, cc:dd] = np.maximum(center[0, aa:bb, cc:dd],
                                                     self.g[a:b, c:d])

                # generate offset (2, h, w) -> (y-dir, x-dir)
                offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0],
                                  mask_index[1])
                offset_x_index = (np.ones_like(mask_index[0]), mask_index[0],
                                  mask_index[1])
                offset[offset_y_index] = center_y - y_coord[mask_index]
                offset[offset_x_index] = center_x - x_coord[mask_index]

        return dict(
            semantic=semantic.astype('long'),
            foreground=foreground.astype('long'),
            center=center.astype(np.float32),
            center_points=center_pts,
            offset=offset.astype(np.float32),
            semantic_weights=semantic_weights.astype(np.float32),
            center_weights=center_weights.astype(np.float32),
            offset_weights=offset_weights.astype(np.float32))


class SemanticTargetGenerator(object):
    """
    Generates semantic training target only for Panoptic-DeepLab (no instance).
    Annotation is assumed to have Cityscapes format.

    Args:
        ignore_index (int): The ignore label for semantic segmentation.
        rgb2id (function): Function, panoptic label is encoded in a colored image, this function convert color to the
            corresponding panoptic label.
    """

    def __init__(self, ignore_index, rgb2id):
        self.ignore_index = ignore_index
        self.rgb2id = rgb2id

    def __call__(self, panoptic, segments):
        """Generates the training target.
        reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py
        reference: https://github.com/facebookresearch/detectron2/blob/master/datasets/prepare_panoptic_fpn.py#L18

        Args:
            panoptic (np.ndarray): Colored image encoding panoptic label.
            segments (list): A list of dictionary containing information of every segment, it has fields:
                - id: panoptic id, after decoding `panoptic`.
                - category_id: semantic class id.
                - area: segment area.
                - bbox: segment bounding box.
                - iscrowd: crowd region.

        Returns:
            A dictionary with fields:
                - semantic: Tensor, semantic label, shape=(H, W).
        """
        panoptic = self.rgb2id(panoptic)
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_index
        for seg in segments:
            cat_id = seg["category_id"]
            semantic[panoptic == seg["id"]] = cat_id

        return dict(semantic=semantic.astype('long'))


class InstanceTargetGenerator(object):
    """
    Generates instance target only for Panoptic-DeepLab.
    Annotation is assumed to have Cityscapes format.

    Args:
        rgb2id (function): Function, panoptic label is encoded in a colored image, this function convert color to the
            corresponding panoptic label.
    """

    def __init__(self, rgb2id):
        self.rgb2id = rgb2id

    def __call__(self, panoptic):
        """Generates the instance target.

        Args:
            panoptic (np.ndarray): Colored image encoding panoptic label.

        Returns:
            A dictionary with fields:
                - instance: Tensor, shape=(H, W). 0 is background. 1, 2, 3 ... is instance, so it is class agnostic.
        """
        panoptic = self.rgb2id(panoptic)
        instance = np.zeros_like(panoptic, dtype=np.int64)
        ids = np.unique(panoptic)
        ins_id = 1
        for i, id in enumerate(ids):
            if id > 1000:
                instance[panoptic == id] = ins_id
                ins_id += 1

        return dict(instance=instance)


class RawPanopticTargetGenerator(object):
    """
    Generator the panoptc ground truth for evaluation, where values are 0,1,2,3,...
        11000, 11001, ..., 18000, 18001, ignore_index(general 255).

    Args:
        ignore_index (int): The ignore label for semantic segmentation.
        rgb2id (function): Function, panoptic label is encoded in a colored image, this function convert color to the
            corresponding panoptic label.
        label_divisor(int, optional): An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id. Default: 1000.
    """

    def __init__(self, ignore_index, rgb2id, label_divisor=1000):
        self.ingore_index = ignore_index
        self.rgb2id = rgb2id
        self.label_divisor = label_divisor

    def __call__(self, panoptic, segments):
        """
        Generates the raw panoptic target

        Args:
            panoptic (numpy.array): colored image encoding panoptic label.
            segments (list): A list of dictionary containing information of every segment, it has fields:
                - id: panoptic id, after decoding `panoptic`.
                - category_id: semantic class id.
                - area: segment area.
                - bbox: segment bounding box.
                - iscrowd: crowd region.

        Returns:
            A dictionary with fields:
                - panoptic: Tensor, panoptic label, shape=(H, W).
        """
        panoptic = self.rgb2id(panoptic)
        raw_panoptic = np.zeros_like(panoptic) + self.ingore_index
        for seg in segments:
            cat_id = seg['category_id']
            # if seg['iscrowd'] == 1:
            #     continue
            if seg['id'] < 1000:
                raw_panoptic[panoptic == seg['id']] = cat_id
            else:
                ins_id = seg['id'] % self.label_divisor
                raw_panoptic[panoptic ==
                             seg['id']] = cat_id * self.label_divisor + ins_id
        return dict(panoptic=raw_panoptic.astype('long'))

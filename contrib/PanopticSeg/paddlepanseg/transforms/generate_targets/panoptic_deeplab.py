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

import numpy as np

from paddlepanseg.cvlibs import manager


@manager.TRANSFORMS.add_component
class GeneratePanopticDeepLabTrainTargets(object):
    def __init__(self,
                 ignore_index,
                 sigma=8,
                 ignore_stuff_in_offset=False,
                 small_instance_area=0,
                 small_instance_weight=1,
                 ignore_crowd_in_semantic=False,
                 num_classes=1):
        self.ignore_index = ignore_index
        self.ignore_stuff_in_offset = ignore_stuff_in_offset
        self.small_instance_area = small_instance_area
        self.small_instance_weight = small_instance_weight
        self.ignore_crowd_in_semantic = ignore_crowd_in_semantic
        self.num_classes = num_classes

        # Generate the default Gaussian image for each center
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def __call__(self, data):
        panoptic = data['label']
        segments_info = data['ann']
        thing_ids = set(data['thing_ids'])
        height, width = panoptic.shape[0], panoptic.shape[1]
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_index
        center = np.zeros((1, height, width), dtype=np.float32)
        center_pts = []
        offset = np.zeros((2, height, width), dtype=np.float32)
        y_coord, x_coord = np.meshgrid(
            np.arange(
                height, dtype=np.float32),
            np.arange(
                width, dtype=np.float32),
            indexing="ij")
        # Generate pixel-wise loss weights
        semantic_weights = np.ones_like(panoptic, dtype=np.uint8)
        # 0: ignore, 1: has instance
        # three conditions for a region to be ignored for instance branches:
        # (1) It is labeled as `ignore_index`
        # (2) It is crowd region (iscrowd=1)
        # (3) (Optional) It is stuff region (for offset branch)
        center_weights = np.zeros_like(panoptic, dtype=np.uint8)
        offset_weights = np.zeros_like(panoptic, dtype=np.uint8)
        for seg in segments_info:
            cat_id = seg['category_id']
            if not (self.ignore_crowd_in_semantic and seg['iscrowd']):
                semantic[panoptic == seg['id']] = cat_id
            if not seg['iscrowd']:
                # Ignored regions are not in `segments_info`.
                # Handle crowd region.
                center_weights[panoptic == seg['id']] = 1
                if not self.ignore_stuff_in_offset or cat_id in thing_ids:
                    offset_weights[panoptic == seg['id']] = 1
            if cat_id in thing_ids:
                # Find instance center
                mask_index = np.where(panoptic == seg['id'])
                if len(mask_index[0]) == 0:
                    # the instance is completely cropped
                    continue

                # Find instance area
                ins_area = len(mask_index[0])
                if ins_area < self.small_instance_area:
                    semantic_weights[panoptic ==
                                     seg['id']] = self.small_instance_weight

                center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[
                    1])
                center_pts.append([center_y, center_x])

                # Generate center heatmap
                y, x = int(round(center_y)), int(round(center_x))
                sigma = self.sigma
                # upper left
                ul = int(np.round(x - 3 * sigma - 1)), int(
                    np.round(y - 3 * sigma - 1))
                # bottom right
                br = int(np.round(x + 3 * sigma + 2)), int(
                    np.round(y + 3 * sigma + 2))

                # Start and end indices in default Gaussian image
                gaussian_x0, gaussian_x1 = max(
                    0, -ul[0]), min(br[0], width) - ul[0]
                gaussian_y0, gaussian_y1 = max(
                    0, -ul[1]), min(br[1], height) - ul[1]

                # Start and end indices in center heatmap image
                center_x0, center_x1 = max(0, ul[0]), min(br[0], width)
                center_y0, center_y1 = max(0, ul[1]), min(br[1], height)
                center[0, center_y0:center_y1, center_x0:center_x1] = np.maximum(
                    center[0, center_y0:center_y1, center_x0:center_x1],
                    self.g[gaussian_y0:gaussian_y1, gaussian_x0:gaussian_x1], )
                offset[0][mask_index] = center_y - y_coord[mask_index]
                offset[1][mask_index] = center_x - x_coord[mask_index]

        center_weights = center_weights[None]
        offset_weights = offset_weights[None]

        data['sem_label'] = semantic.astype('long')
        data['center'] = center.astype('float32')
        data['center_points'] = center_pts
        data['offset'] = offset.astype('float32')
        data['sem_seg_weights'] = semantic_weights.astype('float32')
        data['center_weights'] = center_weights.astype('float32')
        data['offset_weights'] = offset_weights.astype('float32')

        return data

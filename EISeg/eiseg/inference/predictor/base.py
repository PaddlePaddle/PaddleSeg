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
"""
This code is based on https://github.com/saic-vul/ritm_interactive_segmentation
Ths copyright of saic-vul/ritm_interactive_segmentation is as follows:
MIT License [see LICENSE for details]
"""


import paddle
import paddle.nn.functional as F
import numpy as np

from inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide
from .ops import DistMaps, ScaleLayer, BatchImageNormalize


class BasePredictor(object):
    def __init__(
        self,
        model,
        net_clicks_limit=None,
        with_flip=False,
        zoom_in=None,
        max_size=None,
        with_mask=True,
        **kwargs
    ):

        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None
        self.with_prev_mask = with_mask
        self.net = model

        self.normalization = BatchImageNormalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())
        self.dist_maps = DistMaps(
            norm_radius=5, spatial_scale=1.0, cpu_mode=False, use_disks=True
        )

    def to_tensor(self, x):
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                x = x[:, :, None]
        img = paddle.to_tensor(x.transpose([2, 0, 1])).astype("float32") / 255
        return img

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)

        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = paddle.zeros_like(self.original_image[:, :1, :, :])
        if not self.with_prev_mask:
            self.prev_edge = paddle.zeros_like(self.original_image[:, :1, :, :])

    def get_prediction(self, clicker, prev_mask=None):
        clicks_list = clicker.get_clicks()

        input_image = self.original_image
        if prev_mask is None:
            if not self.with_prev_mask:

                prev_mask = self.prev_edge
            else:
                prev_mask = self.prev_prediction

        input_image = paddle.concat([input_image, prev_mask], axis=1)

        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )
        pred_logits, pred_edges = self._get_prediction(
            image_nd, clicks_lists, is_image_changed
        )

        pred_logits = paddle.to_tensor(pred_logits)

        prediction = F.interpolate(
            pred_logits, mode="bilinear", align_corners=True, size=image_nd.shape[2:]
        )
        if pred_edges is not None:
            pred_edge = paddle.to_tensor(pred_edges)
            edge_prediction = F.interpolate(
                pred_edge, mode="bilinear", align_corners=True, size=image_nd.shape[2:]
            )

        for t in reversed(self.transforms):
            if pred_edges is not None:
                edge_prediction = t.inv_transform(edge_prediction)
                self.prev_edge = edge_prediction
            prediction = t.inv_transform(prediction)

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker)

        self.prev_prediction = prediction
        return prediction.numpy()[0, 0]

    def prepare_input(self, image):
        prev_mask = None
        prev_mask = image[:, 3:, :, :]
        image = image[:, :3, :, :]
        image = self.normalization(image)
        return image, prev_mask

    def get_coord_features(self, image, prev_mask, points):

        coord_features = self.dist_maps(image, points)

        if prev_mask is not None:
            coord_features = paddle.concat((prev_mask, coord_features), axis=1)

        return coord_features

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        input_names = self.net.get_input_names()
        self.input_handle_1 = self.net.get_input_handle(input_names[0])
        self.input_handle_2 = self.net.get_input_handle(input_names[1])
        points_nd = self.get_points_nd(clicks_lists)

        image, prev_mask = self.prepare_input(image_nd)
        coord_features = self.get_coord_features(image, prev_mask, points_nd)
        image = image.numpy().astype("float32")
        coord_features = coord_features.numpy().astype("float32")

        self.input_handle_1.copy_from_cpu(image)
        self.input_handle_2.copy_from_cpu(coord_features)

        self.net.run()

        output_names = self.net.get_output_names()

        output_handle = self.net.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()
        if len(output_names) == 3:
            edge_handle = self.net.get_output_handle(output_names[2])
            edge_data = edge_handle.copy_to_cpu()
            return output_data, edge_data
        else:
            return output_data, None

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [
            sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists
        ]
        num_neg_clicks = [
            len(clicks_list) - num_pos
            for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)
        ]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[: self.net_clicks_limit]
            pos_clicks = [
                click.coords_and_indx for click in clicks_list if click.is_positive
            ]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [
                (-1, -1, -1)
            ]

            neg_clicks = [
                click.coords_and_indx for click in clicks_list if not click.is_positive
            ]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [
                (-1, -1, -1)
            ]
            total_clicks.append(pos_clicks + neg_clicks)

        return paddle.to_tensor(total_clicks)

    def get_states(self):
        return {
            "transform_states": self._get_transform_states(),
            "prev_prediction": self.prev_prediction,
        }

    def set_states(self, states):
        self._set_transform_states(states["transform_states"])
        self.prev_prediction = states["prev_prediction"]


def split_points_by_order(tpoints, groups):
    points = tpoints.numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32) for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (
                group_id == 0 and is_negative
            ):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [paddle.to_tensor(x, dtype=tpoints.dtype) for x in group_points]

    return group_points

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

from collections import Counter

import paddle
import paddle.nn.functional as F

from paddlepanseg.cvlibs import build_info_dict, manager
from paddlepanseg.utils import encode_pan_id
from paddlepanseg.postprocessors.base_pp import Postprocessor


@manager.POSTPROCESSORS.add_component
class PanopticDeepLabPostprocessor(Postprocessor):
    def __init__(self,
                 num_classes,
                 thing_ids,
                 stuff_area=2048,
                 threshold=0.1,
                 nms_kernel=7,
                 top_k=200,
                 label_divisor=1000,
                 ignore_index=255):
        super().__init__(
            num_classes=num_classes,
            thing_ids=thing_ids,
            label_divisor=label_divisor,
            ignore_index=ignore_index)
        self.stuff_area = stuff_area
        self.threshold = threshold
        self.nms_kernel = nms_kernel
        self.top_k = top_k

    def _process(self, sample_dict, net_out_dict):
        r = net_out_dict['sem_out'].squeeze(0)
        c = net_out_dict['center'].squeeze(0)
        o = net_out_dict['offset'].squeeze(0)

        if r.ndim != 3:
            raise ValueError

        if c.ndim != 3:
            raise ValueError

        if o.ndim != 3:
            raise ValueError

        pp_out = build_info_dict(_type_='pp_out')

        # For semantic segmentation evaluation.
        sem_prob = F.softmax(r, axis=0)  # [C, H, W]
        pp_out['sem_prob'] = sem_prob.unsqueeze(0)
        pp_out['sem_pred'] = paddle.argmax(
            sem_prob, axis=0, keepdim=True).unsqueeze(0)

        # Post-processing to get panoptic segmentation.
        panoptic_image = self._get_panoptic_segmentation(pp_out['sem_pred'], c,
                                                         o)
        pp_out['pan_pred'] = panoptic_image
        return pp_out

    def _find_instance_center(self, center_heatmap):
        # Thresholding, setting values below threshold to -1.
        center_heatmap[center_heatmap < self.threshold] = -1

        # NMS
        nms_padding = (self.nms_kernel - 1) // 2
        center_heatmap = center_heatmap.unsqueeze(0)
        center_heatmap_max_pooled = F.max_pool2d(
            center_heatmap,
            kernel_size=self.nms_kernel,
            stride=1,
            padding=nms_padding)
        center_heatmap[center_heatmap != center_heatmap_max_pooled] = -1

        center_heatmap = center_heatmap.squeeze()

        # Find non-zero elements.
        if self.top_k is None:
            return paddle.nonzero(center_heatmap > 0)
        else:
            # find top k centers.
            top_k_scores, _ = paddle.topk(
                paddle.flatten(center_heatmap), self.top_k)
            return paddle.nonzero(center_heatmap > paddle.clip(
                top_k_scores[-1], min=0))

    def _group_pixels(self, center_points, offsets, sem_seg):
        height, width = offsets.shape[1:]

        # Generates a coordinate map, where each location is the coordinate of that location.
        y_coord, x_coord = paddle.meshgrid(
            paddle.arange(
                height, dtype=offsets.dtype),
            paddle.arange(
                width, dtype=offsets.dtype), )
        coord = paddle.concat(
            (y_coord.unsqueeze(0), x_coord.unsqueeze(0)), axis=0)

        center_loc = coord + offsets
        center_loc = center_loc.flatten(1).T.unsqueeze_(0)  # [1, H*W, 2]

        center_points = center_points.unsqueeze(1)
        distance = paddle.norm(
            (center_points - center_loc).astype('float32'), axis=-1)
        instance_id = paddle.argmin(
            distance, axis=0).reshape((1, height, width)) + 1

        return instance_id

    def _get_instance_segmentation(self, sem_seg, center_heatmap, offsets,
                                   thing_seg):
        center_points = self._find_instance_center(center_heatmap)
        if center_points.shape[0] == 0:
            return paddle.zeros_like(sem_seg), center_points.unsqueeze(0)
        ins_seg = self._group_pixels(center_points, offsets, sem_seg)
        return thing_seg * ins_seg, center_points.unsqueeze(0)

    def _merge_semantic_and_instance(self, sem_seg, ins_seg,
                                     semantic_thing_seg):
        # In case thing mask does not align with semantic prediction.
        pan_seg = paddle.zeros_like(sem_seg)
        is_thing = (ins_seg > 0) & (semantic_thing_seg > 0)

        # Keep track of instance id for each class.
        class_id_tracker = Counter()

        # Paste thing by majority voting.
        instance_ids = paddle.unique(ins_seg)
        instance_ids = instance_ids[instance_ids != 0]
        for ins_id in instance_ids:
            # Make sure only do majority voting within `semantic_thing_seg`.
            thing_mask = (ins_seg == ins_id) & is_thing
            if paddle.nonzero(thing_mask).shape[0] != 0:
                class_id, _ = paddle.mode(sem_seg[thing_mask].flatten())
                class_id = int(class_id)
                class_id_tracker[class_id] += 1
                new_ins_id = class_id_tracker[class_id]
                pan_seg[thing_mask] = paddle.full(
                    [1],
                    encode_pan_id(
                        class_id, self.label_divisor, ins_id=new_ins_id),
                    dtype='int64')

        # Paste stuff to unoccupied area.
        class_ids = paddle.unique(sem_seg)
        for id_ in self.thing_ids:
            if class_ids.shape[0] != 0:
                class_ids = class_ids[class_ids != id_]
        for class_id in class_ids:
            class_id = int(class_id)
            # Calculate stuff area.
            stuff_mask = (sem_seg == class_id) & (ins_seg == 0)
            if stuff_mask.sum() >= self.stuff_area:
                pan_seg[stuff_mask] = paddle.full(
                    [1],
                    encode_pan_id(class_id, self.label_divisor),
                    dtype='int64')

        return pan_seg

    def _get_panoptic_segmentation(self,
                                   sem_seg,
                                   center_heatmap,
                                   offsets,
                                   foreground_mask=None):
        if foreground_mask is not None:
            thing_seg = foreground_mask
        else:
            # Inference from semantic segmentation
            thing_seg = paddle.zeros_like(sem_seg).astype('bool')
            for thing_class in list(self.thing_ids):
                thing_seg |= sem_seg == thing_class
            thing_seg = paddle.cast(thing_seg, paddle.int64)

        instance, center = self._get_instance_segmentation(
            sem_seg, center_heatmap, offsets, thing_seg)
        panoptic = self._merge_semantic_and_instance(sem_seg, instance,
                                                     thing_seg)

        return panoptic

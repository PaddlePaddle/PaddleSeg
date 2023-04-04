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

# Adapted from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
# 
# Original copyright info: 

# Copyright (c) Facebook, Inc. and its affiliates.

from collections import Counter

import paddle
import paddle.nn.functional as F

from paddlepanseg.cvlibs import build_info_dict, manager
from paddlepanseg.utils import encode_pan_id
from paddlepanseg.postprocessors.base_pp import Postprocessor


@manager.POSTPROCESSORS.add_component
class MaskFormerPostprocessor(Postprocessor):
    def __init__(self,
                 num_classes,
                 thing_ids,
                 object_mask_threshold=0.5,
                 overlap_threshold=0.5,
                 label_divisor=1000,
                 ignore_index=255):
        super().__init__(
            num_classes=num_classes,
            thing_ids=thing_ids,
            label_divisor=label_divisor,
            ignore_index=ignore_index)
        self.object_mask_threshold = object_mask_threshold
        self.overlap_threshold = overlap_threshold

    def _process(self, sample_dict, net_out_dict):
        mask_cls = net_out_dict['logits']
        mask_pred = net_out_dict['masks']
        mask_cls = mask_cls[0]
        mask_pred = mask_pred[0]
        sem_prob, sem_pred = self._semantic_inference(mask_cls, mask_pred)
        _, pan_pred = self._panoptic_inference(mask_cls, mask_pred)
        sem_pred = sem_pred.unsqueeze_([0, 1])
        pan_pred = pan_pred.unsqueeze_([0, 1])
        sem_prob = sem_prob.unsqueeze_(0)
        pp_out = build_info_dict(
            _type_='pp_out',
            pan_pred=pan_pred,
            sem_prob=sem_prob,
            sem_pred=sem_pred)
        return pp_out

    def _panoptic_inference(self, mask_cls, mask_pred):
        scores = F.softmax(mask_cls, axis=1)
        labels = paddle.argmax(scores, axis=1)
        scores = paddle.max(scores, axis=1)
        mask_pred = F.sigmoid(mask_pred)

        h, w = mask_pred.shape[-2:]
        ins_id_map = paddle.zeros((h, w), dtype='int64')
        pan_pred = paddle.zeros((h, w), dtype='int64')

        keep = (labels != self.num_classes) & (
            scores > self.object_mask_threshold)
        keep = paddle.nonzero(keep)

        if keep.shape[0] == 0:
            return ins_id_map, pan_pred

        cur_scores = paddle.index_select(scores, keep, axis=0)
        cur_classes = paddle.index_select(labels, keep, axis=0)
        cur_masks = paddle.index_select(mask_pred, keep, axis=0)
        cur_mask_cls = paddle.index_select(mask_cls, keep, axis=0)
        cur_mask_cls = cur_mask_cls[:, :-1]
        cur_prob_masks = cur_scores.reshape((-1, 1, 1)) * cur_masks

        # Take argmax
        cur_mask_ids = paddle.argmax(cur_prob_masks, axis=0)
        stuff_memory_list = {}
        class_id_tracker = Counter()
        current_segment_id = 0
        for k in range(cur_classes.shape[0]):
            pred_class = int(cur_classes[k])
            isthing = pred_class in self.thing_ids
            mask_area = (cur_mask_ids == k).sum()
            original_area = (cur_masks[k] >= 0.5).sum()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

            if mask_area > 0 and original_area > 0 and mask.sum() > 0:
                if mask_area / original_area < self.overlap_threshold:
                    continue

                # Merge stuff regions
                if not isthing:
                    pan_pred[mask] = encode_pan_id(pred_class,
                                                   self.label_divisor)
                    if pred_class in stuff_memory_list.keys():
                        ins_id_map[mask] = stuff_memory_list[pred_class]
                        continue
                    else:
                        stuff_memory_list[pred_class] = (current_segment_id + 1)
                else:
                    class_id_tracker[pred_class] += 1
                    pan_pred[mask] = paddle.full(
                        [1],
                        encode_pan_id(
                            pred_class,
                            self.label_divisor,
                            ins_id=class_id_tracker[pred_class]),
                        dtype='int64')

                current_segment_id += 1
                ins_id_map[mask] = current_segment_id

        return ins_id_map, pan_pred

    def _semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, axis=1)[..., :-1]
        mask_pred = F.sigmoid(mask_pred)
        sem_prob = paddle.einsum("qc,qhw->chw", mask_cls, mask_pred)
        sem_pred = paddle.argmax(sem_prob, axis=0)
        return sem_prob, sem_pred

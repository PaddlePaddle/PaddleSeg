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

import numpy as np
import paddleseg

import paddlepanseg.transforms.functional as F
from paddlepanseg.cvlibs import manager
from paddlepanseg.cvlibs.info_dicts import InfoDict
from paddlepanseg.utils import encode_pan_id
from .test_transforms import trim_for_test

__all__ = ['Collect', 'ConvertRGBToID', 'DecodeLabels', 'PadToDivisible']


@manager.TRANSFORMS.add_component
class Collect(object):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        if not isinstance(data, InfoDict):
            raise TypeError("data must be an `InfoDict` object.")
        return data.collect(self.keys, return_dict=True)


@manager.TRANSFORMS.add_component
@trim_for_test
class ConvertRGBToID(object):
    def __call__(self, data):
        # Save a copy
        data['label_rgb'] = data['label']
        data['label'] = F.rgb2id(data['label'])
        return data


@manager.TRANSFORMS.add_component
@trim_for_test
class DecodeLabels(object):
    def __init__(self, label_divisor, ignore_index):
        self.label_divisor = label_divisor
        self.ignore_index = ignore_index

    def __call__(self, data):
        raw_label = data['label']
        segments_info = data['ann']
        thing_ids = set(data['thing_ids'])

        ins_label = np.zeros_like(raw_label, dtype='int64')
        sem_label = np.full_like(raw_label, self.ignore_index, dtype='int64')
        pan_label = np.zeros_like(raw_label, dtype='int64')
        ins_id = 0
        class_id_tracker = Counter()

        for seg in segments_info:
            id_ = seg['id']
            mask = (raw_label == id_)
            cat_id = seg['category_id']
            sem_label[mask] = cat_id
            if cat_id in thing_ids:
                if seg['iscrowd'] == 0:
                    # Do not include crowded instances in `ins_label`
                    ins_id += 1
                    ins_label[mask] = ins_id
                # Re-encode `pan_id` using `cat_id` and tracked class instance id
                class_id_tracker[cat_id] += 1
                pan_id = encode_pan_id(
                    cat_id, self.label_divisor, ins_id=class_id_tracker[cat_id])
                pan_label[mask] = pan_id
            else:
                pan_id = encode_pan_id(cat_id, self.label_divisor)
                pan_label[mask] = pan_id
            # Update annotation
            seg['id'] = pan_id

        data['ins_label'] = ins_label
        data['sem_label'] = sem_label
        data['pan_label'] = pan_label

        return data


@manager.TRANSFORMS.add_component
class PadToDivisible(object):
    def __init__(self, size_divisor):
        self.size_divisor = size_divisor

    def __call__(self, data):
        if self.size_divisor > 1:
            h, w = data['img'].shape[:2]
            tar_size = self.get_target_size((w, h))
            if tar_size != (w, h):
                op = paddleseg.transforms.Padding(tar_size)
                data = op(data)
        return data

    def get_target_size(self, size):
        w, h = size
        sd = self.size_divisor
        h = (h + sd - 1) // sd * sd
        w = (w + sd - 1) // sd * sd
        return w, h

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

from paddlepanseg.cvlibs import manager


@manager.TRANSFORMS.add_component
class GenerateMaskFormerTrainTargets(object):
    def __call__(self, data):
        raw_label = data['label']
        segments_info = data['ann']

        gt_ids = []
        gt_masks = []
        for seg in segments_info:
            cat_id = seg['category_id']
            if not seg['iscrowd']:
                gt_ids.append(cat_id)
                gt_masks.append(raw_label == seg['id'])

        data['gt_ids'] = gt_ids
        data['gt_masks'] = gt_masks

        return data

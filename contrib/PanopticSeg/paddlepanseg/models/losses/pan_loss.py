# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.nn as nn


class PanLoss(nn.Layer):
    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, sample_dict, net_out_dict):
        loss = self._compute(sample_dict, net_out_dict)
        return loss

    def _compute(self, sample_dict, net_out_dict):
        raise NotImplementedError


class AdaptedSegLoss(PanLoss):
    def __init__(self,
                 seg_loss_cls,
                 *args,
                 key_maps=None,
                 ignore_index=255,
                 **kwargs):
        super().__init__(ignore_index=ignore_index)
        if key_maps is None:
            key_maps = {
                'logit': ['net_out', 'sem_out'],
                'label': ['sample', 'sem_label']
            }
        self.key_maps = key_maps
        self._loss = seg_loss_cls(*args, **kwargs)

    def _compute(self, sample_dict, net_out_dict):
        kwargs = dict()
        for out_key, (dict_type, in_key) in self.key_maps.items():
            if dict_type == 'sample':
                dict_ = sample_dict
            elif dict_type == 'net_out':
                dict_ = net_out_dict
            else:
                raise ValueError
            kwargs[out_key] = dict_[in_key]
        return self._loss(**kwargs)

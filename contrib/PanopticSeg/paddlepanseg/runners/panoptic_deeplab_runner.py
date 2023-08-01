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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlepanseg.cvlibs import manager
from paddlepanseg.core.runner import PanSegRunner


@manager.RUNNERS.add_component
class PanopticDeepLabRunner(PanSegRunner):
    def compute_losses(self, net_out, data):
        cls_critn, center_critn, offset_critn = self.criteria['types']
        cls_coef, center_coef, offset_coef = self.criteria['coef']
        cls_loss = cls_critn(
            net_out['sem_out'],
            data['sem_label'],
            semantic_weights=data['sem_seg_weights'])
        center_loss = center_critn(net_out['center'], data['center'])
        if 'center_weights' in data:
            center_weights = data['center_weights']
            center_loss = center_loss * center_weights
            if center_weights.sum() > 0:
                center_loss = center_loss.sum() / center_weights.sum()
            else:
                center_loss = center_loss.sum() * 0
        offset_loss = offset_critn(net_out['offset'], data['offset'])
        if 'offset_weights' in data:
            offset_weights = data['offset_weights']
            offset_loss = offset_loss * offset_weights
            if offset_weights.sum() > 0:
                offset_loss = offset_loss.sum() / offset_weights.sum()
            else:
                offset_loss = offset_loss.sum() * 0
        return [
            cls_coef * cls_loss, center_coef * center_loss,
            offset_coef * offset_loss
        ]

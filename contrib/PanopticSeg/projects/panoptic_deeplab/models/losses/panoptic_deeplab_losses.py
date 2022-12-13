# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleseg

from paddlepanseg.cvlibs import manager
from paddlepanseg.models.losses import AdaptedSegLoss, PanLoss

__all__ = ['CenterLoss', 'OffsetLoss', 'CrossEntropyLoss']


@manager.LOSSES.add_component
class CenterLoss(PanLoss):
    """
    Center loss
    """

    def __init__(self, ignore_index=255):
        super().__init__(ignore_index=ignore_index)
        self.criterion = nn.MSELoss(reduction="none")
        # ignore_index unused

    def _compute(self, sample_dict, net_out_dict):
        predictions = net_out_dict['center']
        targets = sample_dict['center']
        weights = sample_dict['center_weights']
        loss = self.criterion(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        return loss


@manager.LOSSES.add_component
class OffsetLoss(PanLoss):
    """
    Offset loss
    """

    def __init__(self, ignore_index=255):
        super().__init__(ignore_index=ignore_index)
        self.criterion = nn.L1Loss(reduction="none")
        # ignore_index unused

    def _compute(self, sample_dict, net_out_dict):
        predictions = net_out_dict['offset']
        targets = sample_dict['offset']
        weights = sample_dict['offset_weights']
        loss = self.criterion(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        return loss


@manager.LOSSES.add_component
class CrossEntropyLoss(AdaptedSegLoss):
    def __init__(self, *args, ignore_index=255, **kwargs):
        key_maps = {
            'logit': ['net_out', 'sem_out'],
            'label': ['sample', 'sem_label'],
            'semantic_weights': ['sample', 'sem_seg_weights']
        }
        super().__init__(
            paddleseg.models.losses.CrossEntropyLoss,
            *args,
            key_maps=key_maps,
            ignore_index=ignore_index,
            **kwargs)

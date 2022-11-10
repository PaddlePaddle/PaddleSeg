# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class MaskFormerLoss(nn.Layer):
    def __init__(self, loss=("labels", 'masks'), ignore_index=255):
        super().__init__()
        mask_weight = 20.0
        dice_weight = 1.0

        weight_dict = {
            "loss_ce": 1,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight
        }
        dec_layers = 6
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update(
                {k + f"_{i}": v
                 for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.weight_dict = weight_dict
        # self.matcher = HungarianMatcher(  #TODO
        #     cost_class=1,
        #     cost_mask=mask_weight,
        #     cost_dice=dice_weight, )

    def forward(self, logit, targets):
        logits_without_aux = {
            k: v
            for k, v in logit.items() if k != "aux_outputs"
        }
        #TODO: add loss
        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(logits_without_aux, targets)

        losses = {'loss_ce': 0.0, "loss_mask": 0.0, "loss_dice": 0.0}

        for k in list(losses.keys()):
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses

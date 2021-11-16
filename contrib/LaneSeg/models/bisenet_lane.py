# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers
from paddleseg.models import BiSeNetV2


@manager.MODELS.add_component
class BiSeNetLane(nn.Layer):
    """
    The BiSeNetLane use BiseNet V2 to process lane detection .

    Args:
        num_classes (int): The unique number of target classes.
        lambd (float, optional): A factor for controlling the size of semantic branch channels. Default: 0.25.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 lambd=0.25,
                 align_corners=False,
                 pretrained=None):
        super().__init__()

        self.net = BiSeNetV2(
            num_classes=num_classes,
            lambd=lambd,
            align_corners=align_corners,
            pretrained=pretrained)

    def forward(self, x):
        logit_list = self.net(x)
        if self.net.training:
            logit_list = [logit_list[0]]
        return logit_list

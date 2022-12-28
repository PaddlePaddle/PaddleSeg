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

import paddleseg

from paddlepanseg.cvlibs import manager
from paddlepanseg.models.losses import AdaptedSegLoss

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

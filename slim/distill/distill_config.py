# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddleslim.dygraph.dist import AdaptorBase


def prepare_distill_adaptor():
    """
    Prepare the distill adaptors for student and teacher model.
    The adaptors set the intermediate feature tensors that used for distillation.
    """

    class StudentAdaptor(AdaptorBase):
        def mapping_layers(self):
            mapping_layers = {}
            # mapping_layers['hidden_0'] = 'layer_name'
            if self.add_tensor:
                # mapping_layers["hidden_0"] = self.model.logit_list
                pass
            return mapping_layers

    class TeatherAdaptor(AdaptorBase):
        def mapping_layers(self):
            mapping_layers = {}
            # mapping_layers['hidden_0'] = 'layer_name'
            if self.add_tensor:
                # mapping_layers["hidden_0"] = self.model.logit_list
                pass
            return mapping_layers

    return StudentAdaptor, TeatherAdaptor


def prepare_distill_config():
    """
    Prepare the distill config.
    """
    '''
    distill_config = [{
        's_feature_idx': 0,
        't_feature_idx': 0,
        'feature_type': 'hidden',
        'loss_function': 'SegChannelwiseLoss',
        'weight': 1.0
    }]
    '''
    distill_config = {}
    return distill_config

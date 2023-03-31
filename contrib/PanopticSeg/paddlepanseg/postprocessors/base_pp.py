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

import abc

from paddlepanseg.cvlibs import build_info_dict


class Postprocessor(metaclass=abc.ABCMeta):
    def __init__(self, num_classes, thing_ids, label_divisor, ignore_index):
        super().__init__()
        self.num_classes = num_classes
        self.thing_ids = thing_ids
        self.label_divisor = label_divisor
        self.ignore_index = ignore_index

    @abc.abstractmethod
    def _process(self, sample_dict, net_out_dict):
        pass

    def __call__(self, sample_dict, net_out_dict):
        pp_out_dict = self._process(sample_dict, net_out_dict)
        return build_info_dict('pp_out', pp_out_dict)

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import yaml
from paddleseg.utils import NoAliasDumper
from paddleseg.cvlibs.config import parse_from_yaml, merge_config_dicts

from .paddle_uapi.config import Config


class SegConfig(Config):
    def update(self, dict_like_obj):
        dict_ = merge_config_dicts(dict_like_obj, self.dict)
        self.reset_from_dict(dict_)

    def load(self, config_file_path):
        dict_ = parse_from_yaml(config_file_path)
        if not isinstance(dict_, dict):
            raise TypeError
        self.reset_from_dict(dict_)

    def dump(self, config_file_path):
        with open(config_file_path, 'w') as f:
            yaml.dump(self.dict, f, Dumper=NoAliasDumper)

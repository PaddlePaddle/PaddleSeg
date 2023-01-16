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

import codecs
import os
from typing import Any, Dict, Generic
import warnings
from ast import literal_eval

import paddle
import yaml
import six

import paddleseg
from paddleseg.utils import logger
from paddleseg.cvlibs import config_checker as checker
from paddleseg.utils.utils import CachedProperty as cached_property


class Config(paddleseg.cvlibs.Config):
    @classmethod
    def _build_default_checker(cls):
        rules = []
        return checker.ConfigChecker(rules, allow_update=True)


class MatBuilder(paddleseg.cvlibs.SegBuilder):
    """
    This class is responsible for building components for matting. 
    """

    @cached_property
    def model(self) -> paddle.nn.Layer:
        model_cfg = self.config.model_cfg
        assert model_cfg != {}, \
            'No model specified in the configuration file.'
        self.show_msg('model', model_cfg)
        return self.build_component(model_cfg)

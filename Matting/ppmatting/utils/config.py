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


class Config(paddleseg.cvlibs.Config):
    def check_sync_config(self) -> None:
        """
        Overload the config due to some checks is not need in matting project.
        """
        if self.dic.get('model', None) is None:
            raise RuntimeError('No model specified in the configuration file.')
        if (not self.train_dataset_config) and (not self.val_dataset_config):
            raise ValueError('One of `train_dataset` or `val_dataset '
                             'should be given, but there are none.')

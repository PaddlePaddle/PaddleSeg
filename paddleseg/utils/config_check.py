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

import numpy as np


def config_check(cfg):
    """
    To check config。

    Args:
        cfg (paddleseg.cvlibs.Config): An object of paddleseg.cvlibs.Config.
    """

    num_classes_check(cfg)


def num_classes_check(cfg):
    """"
    Check that the num_classes in model, train_dataset and val_dataset is consistent.
    """
    num_classes_set = set()
    if cfg.train_dataset and hasattr(cfg.train_dataset, 'num_classes'):
        num_classes_set.add(cfg.train_dataset.num_classes)
    if cfg.val_dataset and hasattr(cfg.val_dataset, 'num_classes'):
        num_classes_set.add(cfg.val_dataset.num_classes)
    if cfg.dic.get('model', None) and cfg.dic['model'].get('num_classes', None):
        num_classes_set.add(cfg.dic['model'].get('num_classes'))
    if (not cfg.train_dataset) and (not cfg.val_dataset):
        raise ValueError(
            'One of `train_dataset` or `val_dataset should be given, but there are none.'
        )
    if len(num_classes_set) == 0:
        raise ValueError(
            '`num_classes` is not found. Please set it in model, train_dataset or val_dataset'
        )
    elif len(num_classes_set) > 1:
        raise ValueError(
            '`num_classes` is not consistent: {}. Please set it consistently in model or train_dataset or val_dataset'
            .format(num_classes_set))

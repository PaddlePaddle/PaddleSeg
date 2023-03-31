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

import copy

import paddleseg
from paddleseg.utils import logger

import paddlepanseg.transforms as T

_ATTR_NAME = 'trim_for_test'


def constr_test_transforms(pipeline, is_list=False):
    if not is_list:
        test_pipeline = copy.deepcopy(pipeline)
        transforms = pipeline.transforms
    else:
        transforms = pipeline
    test_transforms = []
    for tf in transforms:
        if hasattr(tf, _ATTR_NAME) and getattr(tf, _ATTR_NAME) is True:
            continue
        test_transforms.append(tf)
    if isinstance(test_transforms[-1], T.Collect):
        logger.warning("Reset `Collect` to collect input images only.")
        test_transforms[-1] = T.Collect(['img'])
    if not is_list:
        test_pipeline.transforms = test_transforms
        return test_pipeline
    else:
        return test_transforms


def _mark_class_as(cls, tag, val=True):
    if hasattr(cls, tag):
        raise RuntimeError(f"Not possible to mark {cls} as {tag}.")
    setattr(cls, tag, val)
    return cls


def trim_for_test(tf):
    return _mark_class_as(tf, _ATTR_NAME)

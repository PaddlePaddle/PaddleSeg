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
import importlib
from collections.abc import Mapping, Sequence
from contextlib import contextmanager

from tabulate import tabulate
from paddleseg.utils import logger


def set_digits(obj, digits):
    if isinstance(obj, Sequence):
        obj = type(obj)([set_digits(item, digits) for item in obj])
    elif isinstance(obj, Mapping):
        obj = type(obj)(
            {key: set_digits(val, digits)
             for key, val in obj.items()})
    elif isinstance(obj, float):
        obj = round(obj, digits)
    return obj


def tabulate_metrics(metrics,
                     headers=None,
                     newline=True,
                     digits=None,
                     title=None):
    # TODO: Fine granularity of digit control
    if headers is None:
        headers = ["Metric", "Value"]
    metrics = copy.deepcopy(metrics)
    if digits is not None:
        metrics = set_digits(metrics, digits)
    tab = tabulate(metrics.items(), headers=headers)
    if title is not None:
        tab = title + '\n\n' + tab
    if newline:
        tab = '\n' + tab
    return tab


def import_custom_op(op_mod, must, op_sub_mod):
    spec = importlib.util.find_spec(op_mod)
    if spec is None:
        logger.warning(
            f"Custom operator {op_mod} is not found. Please check if it is properly installed."
        )
        if not must:
            logger.warning(f"{op_sub_mod} is used instead of {op_mod}.")
            importlib.import_module(op_sub_mod)
            return mod
        else:
            raise ImportError
    else:
        mod = importlib.import_module(op_mod)
        return mod


@contextmanager
def use_custom_op(op_mod, must=True, op_sub_mod=None):
    mod = import_custom_op(op_mod, must=must, op_sub_mod=op_sub_mod)
    try:
        yield mod
    finally:
        pass

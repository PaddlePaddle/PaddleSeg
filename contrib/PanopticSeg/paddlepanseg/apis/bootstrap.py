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

import importlib


def _load_module_if_exists(module, package=None):
    spec = importlib.util.find_spec(module, package=package)
    if spec is None:
        return None
    mod = importlib.import_module(module, package=package)
    return mod


def load_modules(package=None):
    _load_module_if_exists('datasets', package=package)
    _load_module_if_exists('models', package=package)
    _load_module_if_exists('postprocessors', package=package)
    _load_module_if_exists('transforms', package=package)

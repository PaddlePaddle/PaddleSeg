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

import os.path as osp
from . import core, cvlibs, datasets, models, postprocessors, runners, transforms

__custom_op_path__ = osp.abspath(
    osp.normpath(osp.join(osp.dirname(__file__), 'models', 'ops')))

with open(osp.join(osp.dirname(__file__), ".version"), 'r') as fv:
    __version__ = fv.read().rstrip()

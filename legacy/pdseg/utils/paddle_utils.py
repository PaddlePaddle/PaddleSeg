# coding: utf8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle


def enable_static():
    if hasattr(paddle, 'enable_static'):
        paddle.enable_static()


def save_op_version_info(program_desc):
    if hasattr(paddle.fluid.core, 'save_op_version_info'):
        paddle.fluid.core.save_op_version_info(program_desc)
    else:
        paddle.fluid.core.save_op_compatible_info(program_desc)

# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import os
from paddle import fluid


def load_fp16_vars(executor, dirname, program):
    load_dirname = os.path.normpath(dirname)

    def _if_exist(var):
        name = var.name[:-7] if var.name.endswith('.master') else var.name
        b = os.path.exists(os.path.join(load_dirname, name))
        if not b and isinstance(var, fluid.framework.Parameter):
            print("===== {} not found ====".format(var.name))
        return b

    load_prog = fluid.Program()
    load_block = load_prog.global_block()
    vars = list(filter(_if_exist, program.list_vars()))

    for var in vars:
        new_var = fluid.io._clone_var_in_block_(load_block, var)
        name = var.name[:-7] if var.name.endswith('.master') else var.name
        file_path = os.path.join(load_dirname, name)
        load_block.append_op(
            type='load',
            inputs={},
            outputs={'Out': [new_var]},
            attrs={
                'file_path': file_path,
                'load_as_fp16': var.dtype == fluid.core.VarDesc.VarType.FP16
            })

    executor.run(load_prog)

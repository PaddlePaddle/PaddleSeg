# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

dtype_map = {
    paddle.fluid.core.VarDesc.VarType.FP32: "float32",
    paddle.fluid.core.VarDesc.VarType.FP64: "float64",
    paddle.fluid.core.VarDesc.VarType.FP16: "float16",
    paddle.fluid.core.VarDesc.VarType.INT32: "int32",
    paddle.fluid.core.VarDesc.VarType.INT16: "int16",
    paddle.fluid.core.VarDesc.VarType.INT64: "int64",
    paddle.fluid.core.VarDesc.VarType.BOOL: "bool",
    paddle.fluid.core.VarDesc.VarType.INT16: "int16",
    paddle.fluid.core.VarDesc.VarType.UINT8: "uint8",
    paddle.fluid.core.VarDesc.VarType.INT8: "int8",
}


def convert_dtype_to_string(dtype: str) -> paddle.fluid.core.VarDesc.VarType:
    if dtype in dtype_map:
        return dtype_map[dtype]
    raise TypeError("dtype shoule in %s" % list(dtype_map.keys()))


def get_variable_info(var: paddle.static.Variable) -> dict:
    if not isinstance(var, paddle.static.Variable):
        raise TypeError("var shoule be an instance of paddle.static.Variable")

    var_info = {
        'name': var.name,
        'stop_gradient': var.stop_gradient,
        'is_data': var.is_data,
        'error_clip': var.error_clip,
        'type': var.type
    }

    try:
        var_info['dtype'] = convert_dtype_to_string(var.dtype)
        var_info['lod_level'] = var.lod_level
        var_info['shape'] = var.shape
    except:
        pass

    var_info['persistable'] = var.persistable

    return var_info


def convert_syncbn_to_bn(model_filename):
    """
    Since SyncBatchNorm does not have a cpu kernel, when exporting the model, the SyncBatchNorm
    in the model needs to be converted to BatchNorm.
    """

    def _copy_vars_and_ops_in_blocks(from_block: paddle.device.framework.Block,
                                     to_block: paddle.device.framework.Block):
        for var in from_block.vars:
            var = from_block.var(var)
            var_info = copy.deepcopy(get_variable_info(var))
            if isinstance(var, paddle.device.framework.Parameter):
                to_block.create_parameter(**var_info)
            else:
                to_block.create_var(**var_info)

        for op in from_block.ops:
            all_attrs = op.all_attrs()
            if 'sub_block' in all_attrs:
                _sub_block = to_block.program._create_block()
                _copy_vars_and_ops_in_blocks(all_attrs['sub_block'], _sub_block)
                to_block.program._rollback()
                new_attrs = {'sub_block': _sub_block}
                for key, value in all_attrs.items():
                    if key == 'sub_block':
                        continue
                    new_attrs[key] = copy.deepcopy(value)
            else:
                new_attrs = copy.deepcopy(all_attrs)

            op_type = 'batch_norm' if op.type == 'sync_batch_norm' else op.type
            op_info = {
                'type': op_type,
                'inputs': {
                    input: [
                        to_block._find_var_recursive(var)
                        for var in op.input(input)
                    ]
                    for input in op.input_names
                },
                'outputs': {
                    output: [
                        to_block._find_var_recursive(var)
                        for var in op.output(output)
                    ]
                    for output in op.output_names
                },
                'attrs': new_attrs
            }
            to_block.append_op(**op_info)

    paddle.enable_static()
    with open(model_filename, 'rb') as file:
        desc = file.read()

    origin_program = paddle.static.Program.parse_from_string(desc)
    dest_program = paddle.static.Program()
    _copy_vars_and_ops_in_blocks(origin_program.global_block(),
                                 dest_program.global_block())
    dest_program = dest_program.clone(for_test=True)

    with open(model_filename, 'wb') as file:
        file.write(dest_program.desc.serialize_to_string())

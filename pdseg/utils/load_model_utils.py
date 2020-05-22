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
import os.path as osp

import six
import numpy as np


def parse_param_file(param_file, return_shape=True):
    from paddle.fluid.proto.framework_pb2 import VarType
    f = open(param_file, 'rb')
    version = np.fromstring(f.read(4), dtype='int32')
    lod_level = np.fromstring(f.read(8), dtype='int64')
    for i in range(int(lod_level)):
        _size = np.fromstring(f.read(8), dtype='int64')
        _ = f.read(_size)
    version = np.fromstring(f.read(4), dtype='int32')
    tensor_desc = VarType.TensorDesc()
    tensor_desc_size = np.fromstring(f.read(4), dtype='int32')
    tensor_desc.ParseFromString(f.read(int(tensor_desc_size)))
    tensor_shape = tuple(tensor_desc.dims)
    if return_shape:
        f.close()
        return tuple(tensor_desc.dims)
    if tensor_desc.data_type != 5:
        raise Exception(
            "Unexpected data type while parse {}".format(param_file))
    data_size = 4
    for i in range(len(tensor_shape)):
        data_size *= tensor_shape[i]
    weight = np.fromstring(f.read(data_size), dtype='float32')
    f.close()
    return np.reshape(weight, tensor_shape)


def load_pdparams(exe, main_prog, model_dir):
    import paddle.fluid as fluid
    from paddle.fluid.proto.framework_pb2 import VarType
    from paddle.fluid.framework import Program

    vars_to_load = list()
    vars_not_load = list()
    import pickle
    with open(osp.join(model_dir, 'model.pdparams'), 'rb') as f:
        params_dict = pickle.load(f) if six.PY2 else pickle.load(
            f, encoding='latin1')
    unused_vars = list()
    for var in main_prog.list_vars():
        if not isinstance(var, fluid.framework.Parameter):
            continue
        if var.name not in params_dict:
            print("{} is not in saved model".format(var.name))
            vars_not_load.append(var.name)
            continue
        if var.shape != params_dict[var.name].shape:
            unused_vars.append(var.name)
            vars_not_load.append(var.name)
            print(
                "[SKIP] Shape of pretrained weight {} doesn't match.(Pretrained: {}, Actual: {})"
                .format(var.name, params_dict[var.name].shape, var.shape))
            continue
        vars_to_load.append(var)
    for var_name in unused_vars:
        del params_dict[var_name]
    fluid.io.set_program_state(main_prog, params_dict)

    if len(vars_to_load) == 0:
        print(
            "There is no pretrain weights loaded, maybe you should check you pretrain model!"
        )
    else:
        print("There are {}/{} varaibles in {} are loaded.".format(
            len(vars_to_load),
            len(vars_to_load) + len(vars_not_load), model_dir))


def load_pretrained_weights(exe, main_prog, weights_dir):
    if not osp.exists(weights_dir):
        raise Exception("Path {} not exists.".format(weights_dir))
    if osp.exists(osp.join(weights_dir, "model.pdparams")):
        return load_pdparams(exe, main_prog, weights_dir)
    import paddle.fluid as fluid
    vars_to_load = list()
    vars_not_load = list()
    for var in main_prog.list_vars():
        if not isinstance(var, fluid.framework.Parameter):
            continue
        if not osp.exists(osp.join(weights_dir, var.name)):
            print("[SKIP] Pretrained weight {}/{} doesn't exist".format(
                weights_dir, var.name))
            vars_not_load.append(var)
            continue
        pretrained_shape = parse_param_file(osp.join(weights_dir, var.name))
        actual_shape = tuple(var.shape)
        if pretrained_shape != actual_shape:
            print(
                "[SKIP] Shape of pretrained weight {}/{} doesn't match.(Pretrained: {}, Actual: {})"
                .format(weights_dir, var.name, pretrained_shape, actual_shape))
            vars_not_load.append(var)
            continue
        vars_to_load.append(var)

    params_dict = fluid.io.load_program_state(
        weights_dir, var_list=vars_to_load)
    fluid.io.set_program_state(main_prog, params_dict)

    if len(vars_to_load) == 0:
        print(
            "There is no pretrain weights loaded, maybe you should check you pretrain model!"
        )
    else:
        print("There are {}/{} varaibles in {} are loaded.".format(
            len(vars_to_load),
            len(vars_to_load) + len(vars_not_load), weights_dir))

# coding: utf8
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

import os
import os.path as osp
import numpy as np
import six
import math
from . import logging


def seconds_to_hms(seconds):
    h = math.floor(seconds / 3600)
    m = math.floor((seconds - h * 3600) / 60)
    s = int(seconds - h * 3600 - m * 60)
    hms_str = "{}:{}:{}".format(h, m, s)
    return hms_str


def setting_environ_flags():
    if 'FLAGS_eager_delete_tensor_gb' not in os.environ:
        os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'
    if 'FLAGS_allocator_strategy' not in os.environ:
        os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        if os.environ["CUDA_VISIBLE_DEVICES"].count("-1") > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""


def get_environ_info():
    setting_environ_flags()
    import paddle.fluid as fluid
    info = dict()
    info['place'] = 'cpu'
    info['num'] = int(os.environ.get('CPU_NUM', 1))
    if os.environ.get('CUDA_VISIBLE_DEVICES', None) != "":
        if hasattr(fluid.core, 'get_cuda_device_count'):
            gpu_num = 0
            try:
                gpu_num = fluid.core.get_cuda_device_count()
            except:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                pass
            if gpu_num > 0:
                info['place'] = 'cuda'
                info['num'] = fluid.core.get_cuda_device_count()
    return info


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


def fuse_bn_weights(exe, main_prog, weights_dir):
    import paddle.fluid as fluid
    logging.info("Try to fuse weights of batch_norm...")
    bn_vars = list()
    for block in main_prog.blocks:
        ops = list(block.ops)
        for op in ops:
            if op.type == 'affine_channel':
                scale_name = op.input('Scale')[0]
                bias_name = op.input('Bias')[0]
                prefix = scale_name[:-5]
                mean_name = prefix + 'mean'
                variance_name = prefix + 'variance'
                if not osp.exists(osp.join(
                        weights_dir, mean_name)) or not osp.exists(
                            osp.join(weights_dir, variance_name)):
                    logging.info(
                        "There's no batch_norm weight found to fuse, skip fuse_bn."
                    )
                    return

                bias = block.var(bias_name)
                pretrained_shape = parse_param_file(
                    osp.join(weights_dir, bias_name))
                actual_shape = tuple(bias.shape)
                if pretrained_shape != actual_shape:
                    continue
                bn_vars.append(
                    [scale_name, bias_name, mean_name, variance_name])
    eps = 1e-5
    for names in bn_vars:
        scale_name, bias_name, mean_name, variance_name = names
        scale = parse_param_file(
            osp.join(weights_dir, scale_name), return_shape=False)
        bias = parse_param_file(
            osp.join(weights_dir, bias_name), return_shape=False)
        mean = parse_param_file(
            osp.join(weights_dir, mean_name), return_shape=False)
        variance = parse_param_file(
            osp.join(weights_dir, variance_name), return_shape=False)
        bn_std = np.sqrt(np.add(variance, eps))
        new_scale = np.float32(np.divide(scale, bn_std))
        new_bias = bias - mean * new_scale
        scale_tensor = fluid.global_scope().find_var(scale_name).get_tensor()
        bias_tensor = fluid.global_scope().find_var(bias_name).get_tensor()
        scale_tensor.set(new_scale, exe.place)
        bias_tensor.set(new_bias, exe.place)
    if len(bn_vars) == 0:
        logging.info(
            "There's no batch_norm weight found to fuse, skip fuse_bn.")
    else:
        logging.info("There's {} batch_norm ops been fused.".format(
            len(bn_vars)))


def load_pdparams(exe, main_prog, model_dir):
    import paddle.fluid as fluid
    from paddle.fluid.proto.framework_pb2 import VarType
    from paddle.fluid.framework import Program

    vars_to_load = list()
    import pickle
    with open(osp.join(model_dir, 'model.pdparams'), 'rb') as f:
        params_dict = pickle.load(f) if six.PY2 else pickle.load(
            f, encoding='latin1')
    unused_vars = list()
    for var in main_prog.list_vars():
        if not isinstance(var, fluid.framework.Parameter):
            continue
        if var.name not in params_dict:
            raise Exception("{} is not in saved model".format(var.name))
        if var.shape != params_dict[var.name].shape:
            unused_vars.append(var.name)
            logging.warning(
                "[SKIP] Shape of pretrained weight {} doesn't match.(Pretrained: {}, Actual: {})"
                .format(var.name, params_dict[var.name].shape, var.shape))
            continue
        vars_to_load.append(var)
        logging.debug("Weight {} will be load".format(var.name))
    for var_name in unused_vars:
        del params_dict[var_name]
    fluid.io.set_program_state(main_prog, params_dict)

    if len(vars_to_load) == 0:
        logging.warning(
            "There is no pretrain weights loaded, maybe you should check you pretrain model!"
        )
    else:
        logging.info("There are {} varaibles in {} are loaded.".format(
            len(vars_to_load), model_dir))


def load_pretrain_weights(exe, main_prog, weights_dir, fuse_bn=False):
    if not osp.exists(weights_dir):
        raise Exception("Path {} not exists.".format(weights_dir))
    if osp.exists(osp.join(weights_dir, "model.pdparams")):
        return load_pdparams(exe, main_prog, weights_dir)
    import paddle.fluid as fluid
    vars_to_load = list()
    for var in main_prog.list_vars():
        if not isinstance(var, fluid.framework.Parameter):
            continue
        if not osp.exists(osp.join(weights_dir, var.name)):
            logging.debug("[SKIP] Pretrained weight {}/{} doesn't exist".format(
                weights_dir, var.name))
            continue
        pretrained_shape = parse_param_file(osp.join(weights_dir, var.name))
        actual_shape = tuple(var.shape)
        if pretrained_shape != actual_shape:
            logging.warning(
                "[SKIP] Shape of pretrained weight {}/{} doesn't match.(Pretrained: {}, Actual: {})"
                .format(weights_dir, var.name, pretrained_shape, actual_shape))
            continue
        vars_to_load.append(var)
        logging.debug("Weight {} will be load".format(var.name))

    params_dict = fluid.io.load_program_state(
        weights_dir, var_list=vars_to_load)
    fluid.io.set_program_state(main_prog, params_dict)
    if len(vars_to_load) == 0:
        logging.warning(
            "There is no pretrain weights loaded, maybe you should check you pretrain model!"
        )
    else:
        logging.info("There are {} varaibles in {} are loaded.".format(
            len(vars_to_load), weights_dir))
    if fuse_bn:
        fuse_bn_weights(exe, main_prog, weights_dir)

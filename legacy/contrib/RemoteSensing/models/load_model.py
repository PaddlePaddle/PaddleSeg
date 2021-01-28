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

import yaml
import os.path as osp
import six
import copy
from collections import OrderedDict
import paddle.fluid as fluid
from paddle.fluid.framework import Parameter
from utils import logging
import models


def load_model(model_dir):
    if not osp.exists(osp.join(model_dir, "model.yml")):
        raise Exception("There's no model.yml in {}".format(model_dir))
    with open(osp.join(model_dir, "model.yml")) as f:
        info = yaml.load(f.read(), Loader=yaml.Loader)
    status = info['status']

    if not hasattr(models, info['Model']):
        raise Exception("There's no attribute {} in models".format(
            info['Model']))

    model = getattr(models, info['Model'])(**info['_init_params'])
    if status == "Normal":
        startup_prog = fluid.Program()
        model.test_prog = fluid.Program()
        with fluid.program_guard(model.test_prog, startup_prog):
            with fluid.unique_name.guard():
                model.test_inputs, model.test_outputs = model.build_net(
                    mode='test')
        model.test_prog = model.test_prog.clone(for_test=True)
        model.exe.run(startup_prog)
        import pickle
        with open(osp.join(model_dir, 'model.pdparams'), 'rb') as f:
            load_dict = pickle.load(f)
        fluid.io.set_program_state(model.test_prog, load_dict)

    elif status == "Infer":
        [prog, input_names, outputs] = fluid.io.load_inference_model(
            model_dir, model.exe, params_filename='__params__')
        model.test_prog = prog
        test_outputs_info = info['_ModelInputsOutputs']['test_outputs']
        model.test_inputs = OrderedDict()
        model.test_outputs = OrderedDict()
        for name in input_names:
            model.test_inputs[name] = model.test_prog.global_block().var(name)
        for i, out in enumerate(outputs):
            var_desc = test_outputs_info[i]
            model.test_outputs[var_desc[0]] = out
    if 'test_transforms' in info:
        model.test_transforms = build_transforms(info['test_transforms'])
        model.eval_transforms = copy.deepcopy(model.test_transforms)

    if '_Attributes' in info:
        for k, v in info['_Attributes'].items():
            if k in model.__dict__:
                model.__dict__[k] = v

    logging.info("Model[{}] loaded.".format(info['Model']))
    return model


def build_transforms(transforms_info):
    from transforms import transforms as T
    transforms = list()
    for op_info in transforms_info:
        op_name = list(op_info.keys())[0]
        op_attr = op_info[op_name]
        if not hasattr(T, op_name):
            raise Exception(
                "There's no operator named '{}' in transforms".format(op_name))
        transforms.append(getattr(T, op_name)(**op_attr))
    eval_transforms = T.Compose(transforms)
    return eval_transforms

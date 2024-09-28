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

import os
import paddle
import yaml
import json
from copy import deepcopy

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import logger, utils
from paddleseg.utils.save_info import save_model_info, update_train_results
from paddleseg.deploy.export import WrappedModel


def export(args, model=None, save_dir=None, use_ema=False):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(args.config)
    if not model:
        # save model
        builder = SegBuilder(cfg)
        model = builder.model
        if args.model_path is not None:
            state_dict = paddle.load(args.model_path)
            model.set_dict(state_dict)
            logger.info('Loaded trained params successfully.')
        if args.output_op != 'none':
            model = WrappedModel(model, args.output_op)
        utils.show_env_info()
        utils.show_cfg_info(cfg)
    else:
        pdx_model_name = cfg.dic.get("pdx_model_name", None)
        model = deepcopy(model)
        if args.output_op != 'none' and pdx_model_name != "STFPM":
            model = WrappedModel(model, args.output_op)
    if save_dir is None:
        save_dir = args.save_dir
    os.environ['PADDLESEG_EXPORT_STAGE'] = 'True'

    shape = [None, 3, None, None] if args.input_shape is None \
        else args.input_shape
    input_spec = [paddle.static.InputSpec(shape=shape, dtype='float32')]
    model.eval()
    model = paddle.jit.to_static(model, input_spec=input_spec)
    uniform_output_enabled = cfg.dic.get('uniform_output_enabled', False)
    if args.for_fd or uniform_output_enabled:
        save_name = 'inference'
        yaml_name = 'inference.yml'
    else:
        save_name = 'model'
        yaml_name = 'deploy.yaml'

    if uniform_output_enabled:
        inference_model_path = os.path.join(save_dir, "inference", save_name)
        yml_file = os.path.join(save_dir, "inference", yaml_name)
        if use_ema:
            inference_model_path = os.path.join(save_dir, "inference_ema",
                                                save_name)
            yml_file = os.path.join(save_dir, "inference_ema", yaml_name)

    else:
        inference_model_path = os.path.join(save_dir, save_name)
        yml_file = os.path.join(save_dir, yaml_name)
    paddle.jit.save(model, inference_model_path)

    # save deploy.yaml
    val_dataset_cfg = cfg.val_dataset_cfg
    assert val_dataset_cfg != {}, 'No val_dataset specified in the configuration file.'
    transforms = val_dataset_cfg.get('transforms', None)
    output_dtype = 'int32' if args.output_op == 'argmax' else 'float32'

    # TODO add test config
    deploy_info = {
        'Deploy': {
            'model': save_name + '.pdmodel',
            'params': save_name + '.pdiparams',
            'transforms': transforms,
            'input_shape': shape,
            'output_op': args.output_op,
            'output_dtype': output_dtype
        }
    }
    if cfg.dic.get("pdx_model_name", None):
        deploy_info["Global"] = {}
        deploy_info["Global"]["model_name"] = cfg.dic["pdx_model_name"]
    if cfg.dic.get("hpi_config_path", None):
        with open(cfg.dic["hpi_config_path"], "r") as fp:
            hpi_config = yaml.load(fp, Loader=yaml.SafeLoader)
        if hpi_config["Hpi"]["backend_config"].get("paddle_tensorrt", None):
            hpi_config["Hpi"]["supported_backends"]["gpu"].remove(
                "paddle_tensorrt")
            del hpi_config['Hpi']['backend_config']['paddle_tensorrt']
        if hpi_config["Hpi"]["backend_config"].get("tensorrt", None):
            hpi_config["Hpi"]["supported_backends"]["gpu"].remove("tensorrt")
            del hpi_config['Hpi']['backend_config']['tensorrt']
        hpi_config["Hpi"]["selected_backends"]["gpu"] = "paddle_infer"
        deploy_info["Hpi"] = hpi_config["Hpi"]
    msg = '\n---------------Deploy Information---------------\n'
    msg += str(yaml.dump(deploy_info))
    logger.info(msg)

    with open(yml_file, 'w') as file:
        yaml.dump(deploy_info, file)

    logger.info(f'The inference model is saved in {save_dir}')

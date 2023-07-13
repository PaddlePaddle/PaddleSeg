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

import argparse
import os
import sys

import paddle
import yaml
import paddleseg
from paddleseg.cvlibs import manager
from paddleseg.utils import logger

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..'))

manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()

import ppmatting
from ppmatting.utils import get_input_spec, Config, MatBuilder


def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    # params of training
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the exported model',
        type=str,
        default='./output')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for export',
        type=str,
        default=None)
    parser.add_argument(
        '--trimap',
        dest='trimap',
        help='Whether to input trimap',
        action='store_true')
    parser.add_argument(
        "--input_shape",
        nargs='+',
        help="Export the model with fixed input shape, such as 1 3 1024 1024.",
        type=int,
        default=None)
    parser.add_argument(
        '--for_fd',
        action='store_true',
        help="Export the model to FD-compatible format.")

    return parser.parse_args()


def main(args):
    assert args.cfg is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(args.cfg)
    builder = MatBuilder(cfg)

    paddleseg.utils.show_env_info()
    paddleseg.utils.show_cfg_info(cfg)
    os.environ['PADDLESEG_EXPORT_STAGE'] = 'True'

    net = builder.model
    net.eval()
    if args.model_path:
        para_state_dict = paddle.load(args.model_path)
        net.set_dict(para_state_dict)
        logger.info('Loaded trained params of model successfully.')

    if args.input_shape is None:
        shape = [None, 3, None, None]
    else:
        shape = args.input_shape
    input_spec = get_input_spec(
        net.__class__.__name__, shape=shape, trimap=args.trimap)

    net = paddle.jit.to_static(net, input_spec=input_spec)
    if args.for_fd:
        save_name = 'inference'
        yaml_name = 'inference.yml'
    else:
        save_name = 'model'
        yaml_name = 'deploy.yaml'
    save_path = os.path.join(args.save_dir, save_name)
    paddle.jit.save(net, save_path)

    yml_file = os.path.join(args.save_dir, yaml_name)
    with open(yml_file, 'w') as file:
        transforms = cfg.val_dataset_cfg.get('transforms', [{
            'type': 'Normalize'
        }])
        data = {
            'Deploy': {
                'transforms': transforms,
                'model': save_name + '.pdmodel',
                'params': save_name + '.pdiparams',
                'input_shape': shape
            },
            'ModelName': net.__class__.__name__
        }
        yaml.dump(data, file)

    logger.info(f'Model is saved in {args.save_dir}.')


if __name__ == '__main__':
    args = parse_args()
    main(args)

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

import yaml
import paddle
from paddleslim import QAT

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import logger, utils
from paddleseg.deploy.export import WrappedModel
from qat_config import quant_config
from qat_train import skip_quant


def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument(
        "--config", help="The config file.", type=str, required=True)
    parser.add_argument(
        '--model_path', help='The path of model for export', type=str)
    parser.add_argument(
        "--input_shape",
        nargs='+',
        help="Export the model with fixed input shape, e.g., `--input_shape 1 3 1024 1024`.",
        type=int,
        default=None)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the exported model',
        type=str,
        default='./output/inference_model')
    parser.add_argument(
        '--output_op',
        choices=['argmax', 'softmax', 'none'],
        default="argmax",
        help="Select which op to be appended to output result, default: argmax")
    parser.add_argument(
        '--without_argmax',
        dest='without_argmax',
        help='Do not add the argmax operation at the end of the network',
        action='store_true')
    parser.add_argument(
        '--with_softmax',
        dest='with_softmax',
        help='Add the softmax operation at the end of the network',
        action='store_true')
    parser.add_argument(
        '--for_fd',
        action='store_true',
        help="Export the model to FD-compatible format.")

    return parser.parse_args()


def main(args):
    os.environ['PADDLESEG_EXPORT_STAGE'] = 'True'
    cfg = Config(args.config)
    builder = SegBuilder(cfg)

    net = builder.model

    skip_quant(net)
    quantizer = QAT(config=quant_config)
    quantizer.quantize(net)
    logger.info('Quantize the model successfully')

    if args.model_path is not None:
        utils.load_entire_model(net, args.model_path)
        logger.info('Loaded trained params of model successfully')

    output_op = args.output_op
    if args.without_argmax:
        logger.warning(
            '`--without_argmax` will be deprecated. Please use `--output_op`.')
        output_op = 'none'
    if args.with_softmax:
        logger.warning(
            '`--with_softmax` will be deprecated. Please use `--output_op`.')
        output_op = 'softmax'

    new_net = net if output_op == 'none' else WrappedModel(net, output_op)

    new_net.eval()
    if args.for_fd:
        save_name = 'inference'
        yaml_name = 'inference.yml'
    else:
        save_name = 'model'
        yaml_name = 'deploy.yaml'
    save_path = os.path.join(args.save_dir, save_name)
    shape = [None, 3, None, None] if args.input_shape is None \
        else args.input_shape
    input_spec = [paddle.static.InputSpec(shape=shape, dtype='float32')]
    quantizer.save_quantized_model(new_net, save_path, input_spec=input_spec)

    yml_file = os.path.join(args.save_dir, yaml_name)
    with open(yml_file, 'w') as file:
        transforms = cfg.val_dataset_cfg.get('transforms', [{
            'type': 'Normalize'
        }])
        data = {
            'Deploy': {
                'transforms': transforms,
                'model': save_name + '.pdmodel',
                'params': save_name + '.pdiparams'
            }
        }
        yaml.dump(data, file)

    logger.info(f'The quantized inference model is saved in {args.save_dir}.')


if __name__ == '__main__':
    args = parse_args()
    main(args)

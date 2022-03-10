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

import argparse
import os
import sys

import paddle

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from paddleseg.cvlibs import manager, Config
from paddleseg.core import evaluate
from paddleseg.utils import get_sys_env, logger, config_check, utils
from qat_config import quant_config
from qat_train import skip_quant

from paddleslim import QAT


def get_test_config(cfg, args):

    test_config = cfg.test_config
    if args.aug_eval:
        test_config['aug_eval'] = args.aug_eval
        test_config['scales'] = args.scales

    if args.flip_horizontal:
        test_config['flip_horizontal'] = args.flip_horizontal

    if args.flip_vertical:
        test_config['flip_vertical'] = args.flip_vertical

    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride

    return test_config


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # params of evaluate
    parser.add_argument("--config",
                        dest="cfg",
                        help="The config file.",
                        default=None,
                        type=str)
    parser.add_argument('--model_path',
                        dest='model_path',
                        help='The path of model for evaluation',
                        type=str,
                        default=None)
    parser.add_argument('--num_workers',
                        dest='num_workers',
                        help='Num workers for data loader',
                        type=int,
                        default=0)

    # augment for evaluation
    parser.add_argument(
        '--aug_eval',
        dest='aug_eval',
        help='Whether to use mulit-scales and flip augment for evaluation',
        action='store_true')
    parser.add_argument('--scales',
                        dest='scales',
                        nargs='+',
                        help='Scales for augment',
                        type=float,
                        default=1.0)
    parser.add_argument('--flip_horizontal',
                        dest='flip_horizontal',
                        help='Whether to use flip horizontally augment',
                        action='store_true')
    parser.add_argument('--flip_vertical',
                        dest='flip_vertical',
                        help='Whether to use flip vertically augment',
                        action='store_true')

    # sliding window evaluation
    parser.add_argument('--is_slide',
                        dest='is_slide',
                        help='Whether to evaluate by sliding window',
                        action='store_true')
    parser.add_argument(
        '--crop_size',
        dest='crop_size',
        nargs=2,
        help=
        'The crop size of sliding window, the first is width and the second is height.',
        type=int,
        default=None)
    parser.add_argument(
        '--stride',
        dest='stride',
        nargs=2,
        help=
        'The stride of sliding window, the first is width and the second is height.',
        type=int,
        default=None)
    parser.add_argument(
        '--data_format',
        dest='data_format',
        help=
        'Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')

    return parser.parse_args()


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    # Only support for the DeepLabv3+ model
    if args.data_format == 'NHWC':
        if cfg.dic['model']['type'] != 'DeepLabV3P':
            raise ValueError(
                'The "NHWC" data format only support the DeepLabV3P model!')
        cfg.dic['model']['data_format'] = args.data_format
        cfg.dic['model']['backbone']['data_format'] = args.data_format
        loss_len = len(cfg.dic['loss']['types'])
        for i in range(loss_len):
            cfg.dic['loss']['types'][i]['data_format'] = args.data_format

    val_dataset = cfg.val_dataset
    if val_dataset is None:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )
    elif len(val_dataset) == 0:
        raise ValueError(
            'The length of val_dataset is 0. Please check if your dataset is valid'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model

    skip_quant(model)
    quantizer = QAT(config=quant_config)
    quantizer.quantize(model)
    logger.info('Quantize the model successfully')

    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    test_config = get_test_config(cfg, args)
    config_check(cfg, val_dataset=val_dataset)

    evaluate(model, val_dataset, num_workers=args.num_workers, **test_config)


if __name__ == '__main__':
    args = parse_args()
    main(args)

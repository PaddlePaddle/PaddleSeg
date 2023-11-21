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

import paddle

from paddleseg.cvlibs import manager, Config, SegBuilder
from paddleseg.core import evaluate
from paddleseg.utils import get_sys_env, logger, utils


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # Common params
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help='The path of trained weights to be loaded for evaluation.',
        type=str)
    parser.add_argument(
        '--num_workers',
        help='Number of workers for data loader. Bigger num_workers can speed up data processing.',
        type=int,
        default=0)
    parser.add_argument(
        '--device',
        help='Set the device place for evaluating model.',
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)

    # Data augment params
    parser.add_argument(
        '--aug_eval',
        help='Whether to use mulit-scales and flip augment for evaluation.',
        action='store_true')
    parser.add_argument(
        '--scales',
        nargs='+',
        help='Scales for data augment.',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        help='Whether to use flip horizontally augment.',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        help='Whether to use flip vertically augment.',
        action='store_true')

    # Sliding window evaluation params
    parser.add_argument(
        '--is_slide',
        help='Whether to evaluate images in sliding window method.',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        nargs=2,
        help='The crop size of sliding window, the first is width and the second is height.'
        'For example, `--crop_size 512 512`',
        type=int)
    parser.add_argument(
        '--stride',
        nargs=2,
        help='The stride of sliding window, the first is width and the second is height.'
        'For example, `--stride 512 512`',
        type=int)

    # Other params
    parser.add_argument(
        '--data_format',
        help='Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')
    parser.add_argument(
        '--auc_roc',
        help='Whether to use auc_roc metric.',
        type=bool,
        default=False)
    parser.add_argument(
        '--opts',
        help='Update the key-value pairs of all options.',
        default=None,
        nargs='+')
    # Set multi-label mode
    parser.add_argument(
        '--use_multilabel',
        action='store_true',
        default=False,
        help='Whether to enable multilabel mode. Default: False.')

    return parser.parse_args()


def merge_test_config(cfg, args):
    test_config = cfg.test_config
    if args.aug_eval:
        test_config['aug_eval'] = args.aug_eval
        test_config['scales'] = args.scales
        test_config['flip_horizontal'] = args.flip_horizontal
        test_config['flip_vertical'] = args.flip_vertical
    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride
    if args.use_multilabel:
        test_config['use_multilabel'] = args.use_multilabel
    return test_config


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(args.config, opts=args.opts)
    builder = SegBuilder(cfg)
    test_config = merge_test_config(cfg, args)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_device(args.device)

    # TODO refactor
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

    model = builder.model
    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained weights successfully.')
    val_dataset = builder.val_dataset

    evaluate(model, val_dataset, num_workers=args.num_workers, **test_config)


if __name__ == '__main__':
    args = parse_args()
    main(args)

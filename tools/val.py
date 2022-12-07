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

from paddleseg.cvlibs import manager, Config
from paddleseg.core import evaluate
from paddleseg.utils import get_sys_env, logger, utils


def parse_args():
    '''
    Some input params of previous val.py are moved to config file.
    Please use `-o` or `--opts` to set these params, such as
    aug_eval, scales and so on.
    '''
    parser = argparse.ArgumentParser(description='Model evaluation')
    parser.add_argument('--config', help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help='The path of model weight for evaluation',
        type=str)
    parser.add_argument(
        '-o',
        '--opts',
        help='Update the key-value pairs in config file. For example, '
        '`--o global.num_workers=3 test.aug_eval=True test.scales=0.75,1.0,1.25 '
        'test.flip_horizontal=True`',
        nargs='+')
    return parser.parse_args()

    #TODO parse unknown params


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config.'
    cfg = Config(args.config, opts=args.opts)

    utils.show_env_info()
    utils.show_cfg_info(cfg)

    utils.set_seed(cfg.global_params('seed'))
    utils.set_device(cfg.global_params('device'))
    '''
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
    '''

    model = cfg.model
    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    evaluate(
        model=model,
        eval_dataset=cfg.val_dataset,
        aug_eval=cfg.test_params('aug_eval'),
        scales=cfg.test_params('scales'),
        flip_horizontal=cfg.test_params('flip_horizontal'),
        flip_vertical=cfg.test_params('flip_vertical'),
        is_slide=cfg.test_params('is_slide'),
        stride=cfg.test_params('stride'),
        crop_size=cfg.test_params('crop_size'),
        precision=cfg.global_params('precision'),
        amp_level=cfg.global_params('amp_level'),
        num_workers=cfg.global_params('num_workers'))


if __name__ == '__main__':
    args = parse_args()
    main(args)

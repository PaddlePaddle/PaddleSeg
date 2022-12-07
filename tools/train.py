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
import random

import paddle
import numpy as np
import cv2

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, utils
from paddleseg.core import train


def parse_args():
    '''
    Some input params of previous train.py are moved to config file.
    Please use `-o` or `--opts` to set these params, such as
    batch_size, use_vdl and so on.
    '''
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('--config', help="The path of config file.", type=str)
    parser.add_argument(
        '-o',
        '--opts',
        help='Update the key-value pairs in config file. For example, '
        '`--o train.use_vdl=True train.save_interval=500 global.save_dir=./output`.',
        nargs='+')
    return parser.parse_args()

    #TODO parse unknown params for compatibility


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config.'
    cfg = Config(args.config, opts=args.opts)

    utils.show_env_info()
    utils.show_cfg_info(cfg)

    utils.set_seed(cfg.global_params('seed'))
    utils.set_device(cfg.global_params('device'))
    ''' TODO
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
    if cfg.global_params('device') == 'gpu' and paddle.distributed.ParallelEnv(
    ).nranks > 1:
        model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    val_dataset = cfg.val_dataset if cfg.train_params("do_eval") else None

    train(
        model=model,
        train_dataset=cfg.train_dataset,
        val_dataset=val_dataset,
        optimizer=cfg.optimizer,
        save_dir=cfg.global_params('save_dir'),
        iters=cfg.train_params('iters'),
        batch_size=cfg.train_params('batch_size'),
        resume_model=cfg.train_params('resume_model'),
        save_interval=cfg.train_params('save_interval'),
        log_iters=cfg.train_params('log_iters'),
        num_workers=cfg.global_params('num_workers'),
        use_vdl=cfg.train_params('use_vdl'),
        losses=cfg.loss,
        keep_checkpoint_max=cfg.train_params('keep_checkpoint_max'),
        test_config=cfg.test_config,
        precision=cfg.global_params('precision'),
        amp_level=cfg.global_params('amp_level'),
        profiler_options=cfg.train_params('profiler_options'),
        to_static_training=cfg.train_params("to_static_training"))


if __name__ == '__main__':
    args = parse_args()
    main(args)

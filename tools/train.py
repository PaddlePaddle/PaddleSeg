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

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import get_sys_env, logger, utils
from paddleseg.core import train


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # Common params
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--device',
        help='Set the device place for training model.',
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the model snapshot.',
        type=str,
        default='./output')
    parser.add_argument(
        '--num_workers',
        help='Number of workers for data loader. Bigger num_workers can speed up data processing.',
        type=int,
        default=0)
    parser.add_argument(
        '--do_eval',
        help='Whether to do evaluation in training.',
        action='store_true')
    parser.add_argument(
        '--use_vdl',
        help='Whether to record the data to VisualDL in training.',
        action='store_true')
    parser.add_argument(
        '--use_ema',
        help='Whether to ema the model in training.',
        action='store_true')

    # Runntime params
    parser.add_argument(
        '--resume_model',
        help='The path of the model to resume training.',
        type=str)
    parser.add_argument('--iters', help='Iterations in training.', type=int)
    parser.add_argument(
        '--batch_size', help='Mini batch size of one gpu or cpu. ', type=int)
    parser.add_argument('--learning_rate', help='Learning rate.', type=float)
    parser.add_argument(
        '--save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=1000)
    parser.add_argument(
        '--log_iters',
        help='Display logging information at every `log_iters`.',
        default=10,
        type=int)
    parser.add_argument(
        '--keep_checkpoint_max',
        help='Maximum number of checkpoints to save.',
        type=int,
        default=5)

    # Other params
    parser.add_argument(
        '--seed',
        help='Set the random seed in training.',
        default=None,
        type=int)
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16"],
        help="Use AMP (Auto mixed precision) if precision='fp16'. If precision='fp32', the training is normal."
    )
    parser.add_argument(
        "--amp_level",
        default="O1",
        type=str,
        choices=["O1", "O2"],
        help="Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, the input \
                data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators \
                parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel \
                and batchnorm. Default is O1(amp).")
    parser.add_argument(
        '--profiler_options',
        type=str,
        help='The option of train profiler. If profiler_options is not None, the train ' \
            'profiler is enabled. Refer to the paddleseg/utils/train_profiler.py for details.'
    )
    parser.add_argument(
        '--data_format',
        help='Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')
    parser.add_argument(
        '--repeats',
        type=int,
        default=1,
        help="Repeat the samples in the dataset for `repeats` times in each epoch."
    )
    parser.add_argument(
        '--opts', help='Update the key-value pairs of all options.', nargs='+')
    # Set multi-label mode
    parser.add_argument(
        '--use_multilabel',
        action='store_true',
        default=False,
        help='Whether to enable multilabel mode. Default: False.')
    parser.add_argument(
        '--to_static_training',
        action='store_true',
        default=None,
        help='Whether to enable to_static in training')

    return parser.parse_args()


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(
        args.config,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size,
        to_static_training=args.to_static_training,
        opts=args.opts)
    builder = SegBuilder(cfg)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_seed(args.seed)
    utils.set_device(args.device)
    utils.set_cv2_num_threads(args.num_workers)

    if args.use_multilabel:
        if 'test_config' not in cfg.dic:
            cfg.dic['test_config'] = {'use_multilabel': True}
        else:
            cfg.dic['test_config']['use_multilabel'] = True

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

    model = utils.convert_sync_batchnorm(builder.model, args.device)

    train_dataset = builder.train_dataset
    # TODO refactor
    if args.repeats > 1:
        train_dataset.file_list *= args.repeats
    val_dataset = builder.val_dataset if args.do_eval else None
    optimizer = builder.optimizer
    loss = builder.loss

    train(
        model,
        train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        save_dir=args.save_dir,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        resume_model=args.resume_model,
        save_interval=args.save_interval,
        log_iters=args.log_iters,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        use_ema=args.use_ema,
        losses=loss,
        keep_checkpoint_max=args.keep_checkpoint_max,
        test_config=cfg.test_config,
        precision=args.precision,
        amp_level=args.amp_level,
        profiler_options=args.profiler_options,
        to_static_training=cfg.to_static_training)


if __name__ == '__main__':
    args = parse_args()
    main(args)

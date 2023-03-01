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
from collections import defaultdict
import random

import numpy as np
import paddle
import paddle.nn as nn
import paddleseg
from paddleseg.cvlibs import manager
from paddleseg.utils import get_sys_env, logger

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..'))

manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()

import ppmatting
from ppmatting.core import train
from ppmatting.utils import Config, MatBuilder


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--iters',
        dest='iters',
        help='iters for training',
        type=int,
        default=None)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=None)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=None)
    parser.add_argument(
        '--opts',
        help='Update the key-value pairs of all options.',
        default=None,
        nargs='+')
    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=1000)
    parser.add_argument(
        '--resume_model',
        dest='resume_model',
        help='The path of resume model',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')
    parser.add_argument(
        '--keep_checkpoint_max',
        dest='keep_checkpoint_max',
        help='Maximum number of checkpoints to save',
        type=int,
        default=5)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)
    parser.add_argument(
        '--do_eval',
        dest='do_eval',
        help='Eval while training',
        action='store_true')
    parser.add_argument(
        '--metrics',
        dest='metrics',
        nargs='+',
        help='The metrics to evaluate, it may be the combination of ("sad", "mse", "grad", "conn")',
        type=str,
        default='sad')
    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='Display logging information at every log_iters',
        default=10,
        type=int)
    parser.add_argument(
        '--use_vdl',
        dest='use_vdl',
        help='Whether to record the data to VisualDL during training',
        action='store_true')
    parser.add_argument(
        '--eval_begin_iters',
        dest='eval_begin_iters',
        help='The iters begin evaluation.',
        default=0,
        type=int)
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Set the random seed during training.',
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
                and batchnorm. Default is O1(amp)")
    parser.add_argument(
        '--profiler_options',
        type=str,
        default=None,
        help='The option of train profiler. If profiler_options is not None, the train ' \
            'profiler is enabled. Refer to the paddleseg/utils/train_profiler.py for details.'
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=1,
        help="Repeat the samples in the dataset for `repeats` times in each epoch."
    )
    parser.add_argument(
        '--device',
        dest='device',
        help='Set the device type, which may be GPU, CPU or XPU.',
        default='gpu',
        type=str)
    return parser.parse_args()


def main(args):
    assert args.cfg is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size,
        opts=args.opts)
    builder = MatBuilder(cfg)

    paddleseg.utils.show_env_info()
    paddleseg.utils.show_cfg_info(cfg)
    paddleseg.utils.set_seed(args.seed)
    paddleseg.utils.set_device(args.device)
    paddleseg.utils.set_cv2_num_threads(args.num_workers)

    model = paddleseg.utils.convert_sync_batchnorm(builder.model, args.device)
    train_dataset = builder.train_dataset
    if args.repeats > 1:
        train_dataset.fg_bg_list *= args.repeats
    val_dataset = builder.val_dataset if args.do_eval else None
    optimizer = builder.optimizer

    if args.seed is not None:
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    train(
        model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        save_interval=args.save_interval,
        log_iters=args.log_iters,
        resume_model=args.resume_model,
        save_dir=args.save_dir,
        eval_begin_iters=args.eval_begin_iters,
        metrics=args.metrics,
        precision=args.precision,
        amp_level=args.amp_level,
        profiler_options=args.profiler_options)


if __name__ == '__main__':
    args = parse_args()
    main(args)

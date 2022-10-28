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
from paddleseg.utils import get_sys_env, logger
from paddleseg.core import train


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--iters',
        dest='iters',
        help='Iterations in training.',
        type=int,
        default=None)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu.',
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
        help='The path of the model to resume.',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot.',
        type=str,
        default='./output')
    parser.add_argument(
        '--keep_checkpoint_max',
        dest='keep_checkpoint_max',
        help='Maximum number of checkpoints to save.',
        type=int,
        default=5)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Number of workers for data loader.',
        type=int,
        default=0)
    parser.add_argument(
        '--do_eval',
        dest='do_eval',
        help='Whether to do evaluation while training.',
        action='store_true')
    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='Display logging information at every `log_iters`.',
        default=10,
        type=int)
    parser.add_argument(
        '--use_vdl',
        dest='use_vdl',
        help='Whether to record the data to VisualDL during training.',
        action='store_true')
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
                and batchnorm. Default is O1(amp).")
    parser.add_argument(
        '--data_format',
        dest='data_format',
        help='Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')
    parser.add_argument(
        '--profiler_options',
        type=str,
        default=None,
        help='The option of train profiler. If profiler_options is not None, the train ' \
            'profiler is enabled. Refer to the paddleseg/utils/train_profiler.py for details.'
    )
    parser.add_argument(
        '--device',
        dest='device',
        help='Device place to be set, which can be gpu, xpu, npu, mlu or cpu.',
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)
    parser.add_argument(
        '--repeats',
        type=int,
        default=1,
        help="Repeat the samples in the dataset for `repeats` times in each epoch."
    )

    return parser.parse_args()


def main(args):
    if args.seed is not None:
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    if args.device == 'gpu' and env_info[
            'Paddle compiled with cuda'] and env_info['GPUs used']:
        place = 'gpu'
    elif args.device == 'xpu' and paddle.is_compiled_with_xpu():
        place = 'xpu'
    elif args.device == 'npu' and paddle.is_compiled_with_npu():
        place = 'npu'
    elif args.device == 'mlu' and paddle.is_compiled_with_mlu():
        place = 'mlu'
    else:
        place = 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    nranks = paddle.distributed.ParallelEnv().nranks
    # Limit cv2 threads if too many subprocesses are spawned.
    # This should reduce resource allocation and thus boost performance.
    if nranks >= 8 and args.num_workers >= 8:
        logger.warning(
            "The number of threads used by OpenCV is set to 1 to improve performance."
        )
        cv2.setNumThreads(1)

    cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size,
        opts=args.opts)
    cfg.check_sync_info()

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

    train_dataset = cfg.train_dataset
    if train_dataset is None:
        raise RuntimeError(
            'The training dataset is not specified in the configuration file.')
    elif len(train_dataset) == 0:
        raise ValueError(
            'The length of train_dataset is 0. Please check if your dataset is valid'
        )

    if args.repeats > 1:
        train_dataset.file_list *= args.repeats

    val_dataset = cfg.val_dataset if args.do_eval else None
    losses = cfg.loss

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    # convert bn to sync_bn if necessary
    if place == 'gpu' and paddle.distributed.ParallelEnv().nranks > 1:
        model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(cfg.model)
    else:
        model = cfg.model

    train(
        model,
        train_dataset,
        val_dataset=val_dataset,
        optimizer=cfg.optimizer,
        save_dir=args.save_dir,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        resume_model=args.resume_model,
        save_interval=args.save_interval,
        log_iters=args.log_iters,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        losses=losses,
        keep_checkpoint_max=args.keep_checkpoint_max,
        test_config=cfg.test_config,
        precision=args.precision,
        amp_level=args.amp_level,
        profiler_options=args.profiler_options,
        to_static_training=cfg.to_static_training)

    logger.warning("This `train.py` will be removed in version 2.8, "
                   "please use `tools/train.py`.")


if __name__ == '__main__':
    args = parse_args()
    main(args)

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

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, config_check
from unstructured_prune_utils import train

# python unstructured_prune_train.py --ratio=0.75 --prune_params_type prune_params_type: conv1x1_only --local_sparsity True --pruning_strategy gmp
# assume inital lr=0.1, lr piecewise_decay, pruning iter lr=0.01, tunning iter lr=[0.001,0.0001]


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of pruning
    parser.add_argument(
        '--pruning_mode',
        dest='pruning_mode',
        help=
        'the pruning mode: whether by ratio or by threshold. Default: ratio',
        type=str,
        default='ratio')
    parser.add_argument(
        '--ratio',
        dest='ratio',
        help=
        'The ratio to set zeros, the smaller part bounded by the ratio will be zeros. Default: 0.55',
        type=float,
        default=0.55)
    parser.add_argument(
        '--threshold',
        dest='threshold',
        help='The threshold to set zeros. Default: 0.01',
        type=float,
        default=0.01)
    parser.add_argument(
        '--prune_params_type',
        dest='prune_params_type',
        help=
        'Which kind of params should be pruned, we only support None (all but norms) and conv1x1_only for now. Default: None',
        type=str,
        default=None)
    parser.add_argument(
        '--local_sparsity',
        dest='local_sparsity',
        help=
        'Whether to prune all the parameter matrix at the same ratio or not. Default: False',
        type=bool,
        default=False)

    parser.add_argument(
        '--pruning_strategy',
        dest='pruning_strategy',
        help=
        'Which training strategy to use in pruning, we only support base and gmp for now. Default: base',
        type=str,
        default='base')
    parser.add_argument(
        '--stable_epochs',
        dest='stable_epochs',
        help=
        "The epoch numbers used to stablize the model before pruning. Default: 0",
        type=int,
        default=0)
    parser.add_argument(
        '--pruning_epochs',
        dest='pruning_epochs',
        help=
        'The epoch numbers used to prune the model by a ratio step. Default: 60',
        type=int,
        default=60)
    parser.add_argument(
        '--tunning_epochs',
        dest='tunning_epochs',
        help=
        'The epoch numbers used to tune the after-pruned models. Default: 60',
        type=int,
        default=60)
    parser.add_argument(
        '--last_epoch',
        dest='last_epoch',
        help="The last epoch we'll train from. Default: -1",
        type=int,
        default=-1)
    parser.add_argument(
        '--pruning_steps',
        dest='pruning_steps',
        help=
        'How many times you want to increase your ratio during training. Default: 100',
        type=int,
        default=100)
    parser.add_argument(
        '--initial_ratio',
        dest='initial_ratio',
        help=
        'The initial pruning ratio used at the start of pruning stage. Default: 0.15',
        type=float,
        default=0.15)

    # params of training
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
        '--seed',
        dest='seed',
        help='Set the random seed during training.',
        default=None,
        type=int)
    parser.add_argument(
        '--fp16', dest='fp16', help='Whther to use amp', action='store_true')
    parser.add_argument(
        '--data_format',
        dest='data_format',
        help=
        'Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')
    parser.add_argument(
        '--profiler_options',
        type=str,
        default=None,
        help='The option of train profiler. If profiler_options is not None, the train ' \
            'profiler is enabled. Refer to the paddleseg/utils/train_profiler.py for details.'
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

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size)

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
    val_dataset = cfg.val_dataset if args.do_eval else None
    losses = cfg.loss

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    config_check(cfg, train_dataset=train_dataset, val_dataset=val_dataset)

    train(
        args,
        cfg.model,
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
        fp16=args.fp16,
        profiler_options=args.profiler_options,
        to_static_training=cfg.to_static_training)


if __name__ == '__main__':
    args = parse_args()
    main(args)

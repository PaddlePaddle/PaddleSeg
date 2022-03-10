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
import random
import os
import sys

import paddle
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, config_check, utils
from paddleseg.core import train
from qat_config import quant_config

from paddleslim import QAT
"""
Apply quantization to segmentation model.
NOTE: Only conv2d and linear in backbone are quantized.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument("--config",
                        dest="cfg",
                        help="The config file.",
                        default=None,
                        type=str)
    parser.add_argument('--iters',
                        dest='iters',
                        help='iters for training',
                        type=int,
                        default=None)
    parser.add_argument('--batch_size',
                        dest='batch_size',
                        help='Mini batch size of one gpu or cpu',
                        type=int,
                        default=None)
    parser.add_argument('--learning_rate',
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
    parser.add_argument('--resume_model',
                        dest='resume_model',
                        help='The path of resume model',
                        type=str,
                        default=None)
    parser.add_argument('--save_dir',
                        dest='save_dir',
                        help='The directory for saving the model snapshot',
                        type=str,
                        default='./output')
    parser.add_argument('--keep_checkpoint_max',
                        dest='keep_checkpoint_max',
                        help='Maximum number of checkpoints to save',
                        type=int,
                        default=5)
    parser.add_argument('--num_workers',
                        dest='num_workers',
                        help='Num workers for data loader',
                        type=int,
                        default=0)
    parser.add_argument('--do_eval',
                        dest='do_eval',
                        help='Eval while training',
                        action='store_true')
    parser.add_argument('--log_iters',
                        dest='log_iters',
                        help='Display logging information at every log_iters',
                        default=10,
                        type=int)
    parser.add_argument(
        '--use_vdl',
        dest='use_vdl',
        help='Whether to record the data to VisualDL during training',
        action='store_true')
    parser.add_argument('--seed',
                        dest='seed',
                        help='Set the random seed during training.',
                        default=None,
                        type=int)

    parser.add_argument('--model_path',
                        dest='model_path',
                        help='The path of pretrained model',
                        type=str,
                        default=None)

    return parser.parse_args()


def skip_quant(model):
    """
    If the model has backbone and head, we skip quantizing the conv2d and linear ops
    that belongs the head.
    """
    if not hasattr(model, 'backbone'):
        logger.info("Quantize all target ops")
        return

    logger.info("Quantize all target ops in backbone")
    for name, cur_layer in model.named_sublayers():
        if isinstance(cur_layer, (paddle.nn.Conv2D, paddle.nn.Linear)) \
            and "backbone" not in name:
            cur_layer.skip_quant = True


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

    cfg = Config(args.cfg,
                 learning_rate=args.learning_rate,
                 iters=args.iters,
                 batch_size=args.batch_size)

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

    model = cfg.model
    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    skip_quant(model)
    quantizer = QAT(config=quant_config)
    quantizer.quantize(model)
    logger.info('Quantize the model successfully')

    train(model,
          train_dataset,
          val_dataset=val_dataset,
          optimizer=cfg.optimizer,
          save_dir=args.save_dir,
          iters=cfg.iters,
          batch_size=cfg.batch_size,
          resume_model=None,
          save_interval=args.save_interval,
          log_iters=args.log_iters,
          num_workers=args.num_workers,
          use_vdl=args.use_vdl,
          losses=losses,
          keep_checkpoint_max=args.keep_checkpoint_max)


if __name__ == '__main__':
    args = parse_args()
    main(args)

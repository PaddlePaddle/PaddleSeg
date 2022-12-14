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
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../')))

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, utils

from distill_utils import distill_train
from distill_config import prepare_distill_adaptor, prepare_distill_config

from paddleslim.dygraph.dist import Distill


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument(
        "--student_config",
        help="The config file of the student model.",
        default=None,
        type=str)
    parser.add_argument(
        "--teather_config",
        help="The config file of the teacher model. Distillation only uses "
        "the model in this config.",
        default=None,
        type=str)

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
        help='The path of resume model for the student model',
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

    return parser.parse_args()


def prepare_envs(args):
    """
    Set random seed and the device.
    """
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


def prepare_config(args):
    """
    Create and check the config of student and teacher model.
    Note: we only use the dataset generated by the student config.
    """
    if args.teather_config is None or args.student_config is None:
        raise RuntimeError('No configuration file specified.')

    t_cfg = Config(args.teather_config)
    s_cfg = Config(
        args.student_config,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size)

    train_dataset = s_cfg.train_dataset
    val_dataset = s_cfg.val_dataset if args.do_eval else None
    if train_dataset is None:
        raise RuntimeError(
            'The training dataset is not specified in the configuration file.')
    elif len(train_dataset) == 0:
        raise ValueError(
            'The length of train_dataset is 0. Please check if your dataset is valid'
        )

    msg = '\n---------------Teacher Config Information---------------\n'
    msg += str(t_cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    msg = '\n---------------Student Config Information---------------\n'
    msg += str(s_cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    return t_cfg, s_cfg, train_dataset, val_dataset


def main(args):

    prepare_envs(args)

    t_cfg, s_cfg, train_dataset, val_dataset = prepare_config(args)

    distill_config = prepare_distill_config()

    s_adaptor, t_adaptor = prepare_distill_adaptor()

    t_model = t_cfg.model
    s_model = s_cfg.model
    t_model.eval()
    s_model.train()

    distill_model = Distill(distill_config, s_model, t_model, s_adaptor,
                            t_adaptor)

    distill_train(
        distill_model=distill_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=s_cfg.optimizer,
        save_dir=args.save_dir,
        iters=s_cfg.iters,
        batch_size=s_cfg.batch_size,
        resume_model=args.resume_model,
        save_interval=args.save_interval,
        log_iters=args.log_iters,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        losses=s_cfg.loss,
        distill_losses=s_cfg.distill_loss,
        keep_checkpoint_max=args.keep_checkpoint_max,
        test_config=s_cfg.test_config, )


if __name__ == '__main__':
    args = parse_args()
    main(args)

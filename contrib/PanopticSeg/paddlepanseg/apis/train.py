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

import paddle
from paddleseg.utils import get_sys_env, logger

from paddlepanseg.core import train
from paddlepanseg.cvlibs import manager, Config


def parse_train_args(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Model training")
    # params of training
    parser.add_argument(
        '--config', dest='cfg', help="Config file.", default=None, type=str)
    parser.add_argument(
        '--iters', help="Iterations for training.", type=int, default=None)
    parser.add_argument(
        '--batch_size',
        help="Mini batch size on each GPU (or on CPU).",
        type=int,
        default=None)
    parser.add_argument(
        '--learning_rate', help="Learning rate.", type=float, default=None)
    parser.add_argument(
        '--save_interval',
        help="How many iters to save a model snapshot once during training.",
        type=int,
        default=1000)
    parser.add_argument(
        '--resume_model',
        help="Path of the model snapshot to resume.",
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        help="Directory to save the model snapshot.",
        type=str,
        default="./output")
    parser.add_argument(
        '--keep_checkpoint_max',
        help="Maximum number of checkpoints to save during training.",
        type=int,
        default=5)
    parser.add_argument(
        '--num_workers',
        help="Number of workers used in data loader.",
        type=int,
        default=0)
    parser.add_argument(
        '--do_eval',
        help="To perform evaluation during training.",
        action='store_true')
    parser.add_argument(
        '--log_iters',
        help="Logs will be displayed at every `log_iters`.",
        default=10,
        type=int)
    parser.add_argument(
        '--use_vdl',
        help="To enable VisualDL during training.",
        action='store_true')
    parser.add_argument(
        '--eval_sem',
        help="To calculate semantic segmentation metrics.",
        action='store_true')
    parser.add_argument(
        '--eval_ins',
        help="To calculate instance segmentation metrics.",
        action='store_true')
    parser.add_argument(
        '--precision',
        default='fp32',
        type=str,
        choices=['fp32', 'fp16'],
        help="Use AMP (auto mixed precision) if `precision`='fp16'.")
    parser.add_argument(
        '--amp_level',
        default='O1',
        type=str,
        choices=["O1", "O2"],
        help="Auto mixed precision level. Accepted values are “O1” and “O2”: On O1 level, the input \
                data type of each operator will be casted according to a white list and a black list. \
                On O2 level, all parameters and input data will be casted to fp16, except those for the \
                operators in the black list, those without the support for fp16 kernel, and those for \
                the batchnorm layers. Default is O1.")
    parser.add_argument(
        '--debug', help="To enable debug mode.", action='store_true')

    return parser.parse_args(*args, **kwargs)


def train_with_args(args):
    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError("No configuration file has been specified.")

    cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size)

    train_dataset = cfg.train_dataset
    if train_dataset is None:
        raise RuntimeError(
            "The training dataset is not specified in the configuration file.")
    elif len(train_dataset) == 0:
        raise ValueError(
            "The length of `train_dataset` is 0. Please check if your dataset is valid."
        )
    val_dataset = cfg.val_dataset if args.do_eval else None
    losses = cfg.loss

    msg = "\n---------------Config Information---------------\n"
    msg += str(cfg)
    msg += "------------------------------------------------"
    logger.info(msg)

    # Convert bn to sync_bn if necessary
    if place == 'gpu' and paddle.distributed.ParallelEnv().nranks > 1:
        model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(cfg.model)
    else:
        model = cfg.model

    try:
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
            postprocessor=cfg.postprocessor,
            eval_sem=args.eval_sem,
            eval_ins=args.eval_ins,
            precision=args.precision,
            amp_level=args.amp_level)
    except BaseException as e:
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
            import sys
            import pdb
            pdb.post_mortem(sys.exc_info()[2])
            exit(1)
        else:
            raise


if __name__ == '__main__':
    args = parse_train_args()
    train_with_args(args)

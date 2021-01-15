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
from functools import partial

import paddle
from paddleslim.dygraph import L1NormFilterPruner
from paddleslim.analysis import dygraph_flops
from paddleseg.cvlibs.config import Config
from paddleseg.core.val import evaluate
from paddleseg.core.train import train
from paddleseg.utils import get_sys_env, logger


def parse_args():
    parser = argparse.ArgumentParser(description='Model pruning')
    # params of pruning
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        "--prune_ratio",
        dest="prune_ratio",
        help="The ratio of model pruning.",
        type=float,
        default=None,
        required=True)
    parser.add_argument(
        '--retraining_iters',
        dest='retraining_iters',
        help='Number of iterations of retraining.',
        type=int,
        default=None,
        required=True)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu.',
        type=int,
        default=None)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate of retraining.',
        type=float,
        default=None)
    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help=
        'The number of interval iterations to save the model during training.',
        type=int,
        default=1000)
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
        help='Eval while retraining',
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

    return parser.parse_args()


def eval_fn(net, eval_dataset):
    miou, _ = evaluate(net, eval_dataset, num_workers=6, print_detail=False)
    return miou


def main(args):
    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'
    paddle.set_device(place)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.retraining_iters,
        batch_size=args.batch_size)

    train_dataset = cfg.train_dataset
    if not train_dataset:
        raise RuntimeError(
            'The training dataset is not specified in the configuration file.')

    val_dataset = cfg.val_dataset
    if not val_dataset:
        raise RuntimeError(
            'The validation dataset is not specified in the configuration file.'
        )
    net = cfg.model

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    logger.info(
        'Step 1/3: Start calculating the sensitivity of model parameters...')
    sample_shape = [1] + list(val_dataset[0][0].shape)
    sen_file = os.path.join(args.save_dir, 'sen.pickle')
    pruner = L1NormFilterPruner(net, sample_shape)
    pruner.sensitive(
        eval_func=partial(eval_fn, net, val_dataset), sen_file=sen_file)
    logger.info(
        f'The sensitivity calculation of model parameters is complete. The result is saved in {sen_file}.'
    )

    flops = dygraph_flops(net, sample_shape)
    logger.info(
        f'Step 2/3: Start to prune the model, the ratio of pruning is {args.pruning_ratio}. FLOPs before pruning: {flops}.'
    )
    pruner.sensitive_prune(args.pruned_flops)
    flops = dygraph_flops(net, sample_shape)
    logger.info(f'Model pruning completed. FLOPs after pruning: {flops}.')

    logger.info(f'Step 3/3: Start retraining the model.')
    train(
        net,
        train_dataset,
        val_dataset=val_dataset,
        optimizer=cfg.optimizer,
        save_dir=args.save_dir,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        save_interval=args.save_interval,
        log_iters=args.log_iters,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        losses=cfg.loss,
        keep_checkpoint_max=args.keep_checkpoint_max)


if __name__ == '__main__':
    args = parse_args()
    main(args)

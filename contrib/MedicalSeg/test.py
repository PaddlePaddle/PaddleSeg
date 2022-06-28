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

from medicalseg.cvlibs import Config
from medicalseg.core import evaluate
from medicalseg.utils import get_sys_env, logger, config_check, utils


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # params of evaluate
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)

    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for evaluation',
        type=str,
        default="saved_model/vnet_lung_coronavirus_128_128_128_15k/best_model/model.pdparams"
    )

    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The path to save result',
        type=str,
        default="saved_model/vnet_lung_coronavirus_128_128_128_15k/best_model")

    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)

    parser.add_argument(
        '--print_detail',  # the dest cannot have space in it
        help='Whether to print evaluate values',
        type=bool,
        default=True)

    parser.add_argument(
        '--use_vdl',
        help='Whether to use visualdl to record result images',
        type=bool,
        default=True)

    parser.add_argument(
        '--auc_roc',
        help='Whether to use auc_roc metric',
        type=bool,
        default=False)

    parser.add_argument('--sw_num', default=None, type=int, help='sw_num')

    parser.add_argument(
        '--is_save_data', default=True, type=eval, help='warmup')

    parser.add_argument(
        '--has_dataset_json', default=True, type=eval, help='has_dataset_json')

    return parser.parse_args()


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    losses = cfg.loss
    test_dataset = cfg.test_dataset

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    if args.use_vdl:
        from visualdl import LogWriter
        log_writer = LogWriter(args.save_dir)

    evaluate(
        model,
        test_dataset,
        losses,
        num_workers=args.num_workers,
        print_detail=args.print_detail,
        auc_roc=args.auc_roc,
        writer=log_writer,
        save_dir=args.save_dir,
        sw_num=args.sw_num,
        is_save_data=args.is_save_data,
        has_dataset_json=args.has_dataset_json)


if __name__ == '__main__':
    args = parse_args()
    main(args)

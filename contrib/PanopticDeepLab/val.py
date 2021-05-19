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

import paddle
import paddleseg
from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, config_check

from core import evaluate
from datasets import CityscapesPanoptic
from models import PanopticDeepLab


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
        default=None)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)
    parser.add_argument(
        '--threshold',
        dest='threshold',
        help='Threshold applied to center heatmap score',
        type=float,
        default=0.1)
    parser.add_argument(
        '--nms_kernel',
        dest='nms_kernel',
        help='NMS max pooling kernel size',
        type=int,
        default=7)
    parser.add_argument(
        '--top_k',
        dest='top_k',
        help='Top k centers to keep',
        type=int,
        default=200)

    return parser.parse_args()


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    val_dataset = cfg.val_dataset
    if val_dataset is None:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )
    elif len(val_dataset) == 0:
        raise ValueError(
            'The length of val_dataset is 0. Please check if your dataset is valid'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    if args.model_path:
        paddleseg.utils.utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    config_check(cfg, val_dataset=val_dataset)

    evaluate(
        model,
        val_dataset,
        threshold=args.threshold,
        nms_kernel=args.nms_kernel,
        top_k=args.top_k,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)

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

from paddleseg.cvlibs import manager, Config
from paddleseg.core.val_img import evaluate
from paddleseg.utils import get_sys_env, logger, utils
from paddleseg.datasets.dataset_no_transform import DatasetNoTransform


def parse_args():
    parser = argparse.ArgumentParser(description='Images evaluation')

    # params of evaluate
    parser.add_argument(
        '--img_dir', dest='img_dir', help='Images directory', type=str)
    parser.add_argument(
        '--label_dir', dest='label_dir', help='Labels directory', type=str)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)
    return parser.parse_args()


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    val_dataset = DatasetNoTransform(args.img_dir, args.label_dir)
    if val_dataset is None:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )

    logger.info('Loaded trained params of model successfully')

    evaluate(
        val_dataset,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)

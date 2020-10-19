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
# from paddleseg.core import evaluate
from paddleseg.core.val_mutiscale_slide import evaluate
from paddleseg.utils import get_sys_env, logger


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # params of evaluate
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='The path of model for evaluation',
        type=str,
        default=None)

    return parser.parse_args()


def main(args):
    env_info = get_sys_env()
    places = paddle.CUDAPlace(paddle.distributed.ParallelEnv().dev_id) \
        if env_info['Paddle compiled with cuda'] and env_info['GPUs used'] \
        else paddle.CPUPlace()

    paddle.disable_static(places)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    val_dataset = cfg.val_dataset
    if not val_dataset:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    ckpt_path = os.path.join(args.model_dir, 'model')
    para_state_dict, opti_state_dict = paddle.load(ckpt_path)
    model.set_dict(para_state_dict)
    logger.info('Loaded trained params of model successfully')

    evaluate(model, val_dataset)


if __name__ == '__main__':
    args = parse_args()
    main(args)

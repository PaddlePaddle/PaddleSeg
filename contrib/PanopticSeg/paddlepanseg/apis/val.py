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
import paddleseg
from paddleseg.utils import get_sys_env, logger

from paddlepanseg.core import evaluate
from paddlepanseg.cvlibs import manager, Config


def parse_val_args(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Model evaluation")

    # params of evaluate
    parser.add_argument(
        "--config", dest="cfg", help="Config file.", default=None, type=str)
    parser.add_argument(
        '--model_path',
        help="Path of the model to evaluate.",
        type=str,
        default=None)
    parser.add_argument(
        '--num_workers',
        help="Number of workers used in data loader.",
        type=int,
        default=0)
    parser.add_argument(
        '--eval_sem',
        help="To calculate semantic segmentation metrics.",
        action='store_true')
    parser.add_argument(
        '--eval_ins',
        help="To calculate instance segmentation metrics.",
        action='store_true')
    parser.add_argument(
        '--debug', help="To enable debug mode.", action='store_true')

    return parser.parse_args(*args, **kwargs)


def val_with_args(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError("No configuration file has been specified.")

    cfg = Config(args.cfg)
    val_dataset = cfg.val_dataset
    if val_dataset is None:
        raise RuntimeError(
            "The validation dataset is not specified in the configuration file.")
    elif len(val_dataset) == 0:
        raise ValueError(
            "The length of `val_dataset` is 0. Please check if your dataset is valid."
        )

    msg = "\n---------------Config Information---------------\n"
    msg += str(cfg)
    msg += "------------------------------------------------"
    logger.info(msg)

    model = cfg.model
    if args.model_path:
        paddleseg.utils.utils.load_entire_model(model, args.model_path)
        logger.info("Params are successfully loaded.")

    try:
        evaluate(
            model,
            val_dataset,
            postprocessor=cfg.postprocessor,
            num_workers=args.num_workers,
            eval_sem=args.eval_sem,
            eval_ins=args.eval_ins)
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
    args = parse_val_args()
    val_with_args(args)

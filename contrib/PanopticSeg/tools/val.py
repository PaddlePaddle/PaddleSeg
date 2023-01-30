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

import paddleseg
from paddleseg.utils import logger, utils

from paddlepanseg.core import evaluate
from paddlepanseg.cvlibs import Config, make_default_builder


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
    parser.add_argument(
        '--device',
        help="Device for evaluating model.",
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)

    return parser.parse_args(*args, **kwargs)


def val_with_args(args):
    if not args.cfg:
        raise RuntimeError("No configuration file has been specified.")
    cfg = Config(args.cfg)
    builder = make_default_builder(cfg)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_device(args.device)

    model = builder.model
    if args.model_path:
        paddleseg.utils.utils.load_entire_model(model, args.model_path)
        logger.info("Params are successfully loaded.")
    val_dataset = builder.val_dataset
    postprocessor = builder.postprocessor
    runner = builder.runner

    try:
        evaluate(
            model,
            val_dataset,
            postprocessor=postprocessor,
            runner=runner,
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

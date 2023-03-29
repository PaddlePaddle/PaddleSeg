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

from paddleseg.utils import logger, utils

from paddlepanseg.core import train
from paddlepanseg.cvlibs import Config, make_default_builder


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
    parser.add_argument(
        '--device',
        help="Device for training model.",
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)
    parser.add_argument(
        '--profiler_options',
        type=str,
        help="Options of the training profiler. If `profiler_options` is not None, the training profiler will be enabled. Refer to `paddleseg/utils/train_profiler.py` for details."
    )
    parser.add_argument('--seed', help="Random seed.", default=None, type=int)
    parser.add_argument('--opts', help="Update the key-value pairs of existing options.", nargs='+')

    return parser.parse_args(*args, **kwargs)


def train_with_args(args):
    if args.cfg is None:
        raise RuntimeError("No configuration file has been specified.")
    cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size,
        opts=args.opts)
    builder = make_default_builder(cfg)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_seed(args.seed)
    utils.set_device(args.device)
    utils.set_cv2_num_threads(args.num_workers)

    model = utils.convert_sync_batchnorm(builder.model, args.device)

    train_dataset = builder.train_dataset
    val_dataset = builder.val_dataset if args.do_eval else None
    losses = builder.loss
    optimizer = builder.optimizer
    postprocessor = builder.postprocessor
    runner = builder.runner

    try:
        train(
            model,
            train_dataset,
            val_dataset=val_dataset,
            optimizer=optimizer,
            postprocessor=postprocessor,
            runner=runner,
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
            eval_sem=args.eval_sem,
            eval_ins=args.eval_ins,
            precision=args.precision,
            amp_level=args.amp_level,
            profiler_options=args.profiler_options,
            to_static_training=cfg.to_static_training)
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

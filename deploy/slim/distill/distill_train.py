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

from paddleslim.dygraph.dist import Distill

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import logger, utils
from distill_utils import distill_train
from distill_config import prepare_distill_adaptor, prepare_distill_config


def parse_args():
    parser = argparse.ArgumentParser(description='Knowledge distillation')

    parser.add_argument(
        '--student_config',
        help="The config file of the student model.",
        default=None,
        type=str)
    parser.add_argument(
        '--teather_config',
        help="The config file of the teacher model.",
        default=None,
        type=str)
    parser.add_argument(
        '--iters',
        help="The total iterations of training.",
        type=int,
        default=None)
    parser.add_argument(
        '--batch_size',
        help="The mini-batch size on each GPU.",
        type=int,
        default=None)
    parser.add_argument(
        '--learning_rate', help="The learning rate.", type=float, default=None)
    parser.add_argument(
        '--save_interval',
        help="How many iters to save a model snapshot during training.",
        type=int,
        default=1000)
    parser.add_argument(
        '--resume_model',
        dest='resume_model',
        help="The path of the snapshot to resume for the student model.",
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        help="The directory for saving model snapshots.",
        type=str,
        default='./output')
    parser.add_argument(
        '--keep_checkpoint_max',
        help="The maximum number of model snapshots to save.",
        type=int,
        default=5)
    parser.add_argument(
        '--num_workers',
        help="The number of worker processes used in the data loader.",
        type=int,
        default=0)
    parser.add_argument(
        '--do_eval',
        help="Whether or not to validate model during training.",
        action='store_true')
    parser.add_argument(
        '--log_iters',
        help="Display logging information every `log_iters` iterations.",
        default=10,
        type=int)
    parser.add_argument(
        '--use_vdl',
        help="Whether or not to enable VisualDL during training.",
        action='store_true')
    parser.add_argument(
        '--seed',
        help="The random seed used in training.",
        default=None,
        type=int)

    return parser.parse_args()


def prepare_envs(args):
    """
    Set random seed and the device.
    """

    utils.set_seed(args.seed)
    utils.show_env_info()

    env_info = utils.get_sys_env()
    place = 'gpu' if env_info['GPUs used'] else 'cpu'
    utils.set_device(place)


def main(args):

    prepare_envs(args)

    if args.teather_config is None or args.student_config is None:
        raise RuntimeError(
            "At least one of the configuration files is not specified.")

    t_cfg = Config(args.teather_config)
    s_cfg = Config(
        args.student_config,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size)
    t_builder = SegBuilder(t_cfg)
    s_builder = SegBuilder(s_cfg)

    msg = "\n---------------Teacher Config Information---------------\n"
    msg += str(t_cfg)
    msg += "------------------------------------------------"
    logger.info(msg)

    msg = "\n---------------Student Config Information---------------\n"
    msg += str(s_cfg)
    msg += "------------------------------------------------"
    logger.info(msg)

    distill_config = prepare_distill_config()

    s_adaptor, t_adaptor = prepare_distill_adaptor()

    t_model = t_builder.model
    s_model = s_builder.model
    t_model.eval()
    s_model.train()

    distill_model = Distill(distill_config, s_model, t_model, s_adaptor,
                            t_adaptor)

    distill_train(
        distill_model=distill_model,
        train_dataset=s_builder.train_dataset,
        val_dataset=s_builder.val_dataset,
        optimizer=s_builder.optimizer,
        save_dir=args.save_dir,
        iters=s_builder.iters,
        batch_size=s_builder.batch_size,
        resume_model=args.resume_model,
        save_interval=args.save_interval,
        log_iters=args.log_iters,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        losses=s_builder.loss,
        distill_losses=s_builder.distill_loss,
        keep_checkpoint_max=args.keep_checkpoint_max,
        test_config=s_cfg.test_cfg, )


if __name__ == '__main__':
    args = parse_args()
    main(args)

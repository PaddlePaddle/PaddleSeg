# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import datasets
import models
from cvlibs import Config, CPSBuilder
from paddleseg.utils import utils
from core import train


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # Common params
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--device',
        help='Set the device place for training model.',
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the model snapshot.',
        type=str,
        default='./output')
    parser.add_argument(
        '--num_workers',
        help='Number of workers for data loader. Bigger num_workers can speed up data processing.',
        type=int,
        default=0)
    parser.add_argument(
        '--do_eval',
        help='Whether to do evaluation in training.',
        action='store_true')
    parser.add_argument(
        '--use_vdl',
        help='Whether to record the data to VisualDL in training.',
        action='store_true')

    # Runntime params
    parser.add_argument(
        '--resume_model',
        help='The path of the model to resume training.',
        type=str)
    parser.add_argument('--nepochs', help='Iterations in training.', type=int)
    parser.add_argument(
        '--batch_size', help='Mini batch size of one gpu or cpu. ', type=int)
    parser.add_argument(
        '--labeled_ratio',
        help='The ratio of total data to labeled data, if 2, is 1/2, i.e. 0.5',
        type=int,
        default=2)
    parser.add_argument('--learning_rate', help='Learning rate.', type=float)
    parser.add_argument(
        '--save_epoch',
        help='How many epochs to save a model snapshot once during training.',
        type=int,
        default=5)
    parser.add_argument(
        '--log_iters',
        help='Display logging information at every `log_iters`.',
        default=10,
        type=int)
    parser.add_argument(
        '--keep_checkpoint_max',
        help='Maximum number of checkpoints to save.',
        type=int,
        default=5)

    # Other params
    parser.add_argument(
        '--seed',
        help='Set the random seed in training.',
        default=None,
        type=int)
    parser.add_argument(
        '--profiler_options',
        type=str,
        help='The option of train profiler. If profiler_options is not None, the train ' \
            'profiler is enabled. Refer to the paddleseg/utils/train_profiler.py for details.'
    )
    parser.add_argument(
        '--data_format',
        help='Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')
    parser.add_argument(
        '--opts', help='Update the key-value pairs of all options.', nargs='+')

    return parser.parse_args()


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(
        args.config,
        learning_rate=args.learning_rate,
        nepochs=args.nepochs,
        batch_size=args.batch_size,
        labeled_ratio=args.labeled_ratio,
        opts=args.opts)
    builder = CPSBuilder(cfg)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_seed(args.seed)
    utils.set_device(args.device)
    utils.set_cv2_num_threads(args.num_workers)

    # TODO refactor
    # Only support for the DeepLabv3+ model
    if args.data_format == 'NHWC':
        if cfg.dic['model']['type'] != 'DeepLabV3P':
            raise ValueError(
                'The "NHWC" data format only support the DeepLabV3P model!')
        cfg.dic['model']['data_format'] = args.data_format
        cfg.dic['model']['backbone']['data_format'] = args.data_format
        loss_len = len(cfg.dic['loss']['types'])
        for i in range(loss_len):
            cfg.dic['loss']['types'][i]['data_format'] = args.data_format

    model = utils.convert_sync_batchnorm(builder.model, args.device)

    train_dataset = builder.train_dataset
    unsupervised_train_dataset = builder.unsupervised_train_dataset
    mask_genarator = builder.batch_transforms

    val_dataset = builder.val_dataset if args.do_eval else None
    optimizer_l = builder.optimizer_l
    optimizer_r = builder.optimizer_r
    loss = builder.loss

    train(
        model,
        train_dataset,
        unsupervised_train_dataset=unsupervised_train_dataset,
        mask_genarator=mask_genarator,
        val_dataset=val_dataset,
        optimizer_l=optimizer_l,
        optimizer_r=optimizer_r,
        save_dir=args.save_dir,
        nepochs=cfg.nepochs,
        labeled_ratio=cfg.labeled_ratio,
        batch_size=cfg.batch_size,
        resume_model=args.resume_model,
        save_epoch=args.save_epoch,
        log_iters=args.log_iters,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        losses=loss,
        keep_checkpoint_max=args.keep_checkpoint_max,
        test_config=cfg.test_config,
        profiler_options=args.profiler_options)


if __name__ == '__main__':
    args = parse_args()
    main(args)

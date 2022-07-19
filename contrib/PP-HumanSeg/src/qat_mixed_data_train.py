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
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import paddle

from paddleseg.cvlibs import manager
from paddleseg.utils import get_sys_env, logger, config_check, utils
from paddleseg.transforms import *
from paddleseg.models import *
from paddleseg.models.losses import *
from paddleseg.datasets import Dataset
from datasets.mixed_dataset import MixedDataset

from paddleslim import QAT

from mixed_data_train_helper import train
from config import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--iters',
        dest='iters',
        help='iters for training',
        type=int,
        default=None)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=None)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=None)
    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=None)
    parser.add_argument(
        '--resume_model',
        dest='resume_model',
        help='The path of resume model',
        type=str,
        default=None)
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
        help='Eval while training',
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

    parser.add_argument(
        '--model_path',
        help='The path of pretrained model',
        type=str,
        default=None)

    return parser.parse_args()


quant_config = {
    # weight preprocess type, default is None and no preprocessing is performed.
    'weight_preprocess_type': None,
    # activation preprocess type, default is None and no preprocessing is performed.
    'activation_preprocess_type': None,
    # weight quantize type, default is 'channel_wise_abs_max'
    'weight_quantize_type': 'channel_wise_abs_max',
    # activation quantize type, default is 'moving_average_abs_max'
    'activation_quantize_type': 'moving_average_abs_max',
    # weight quantize bit num, default is 8
    'weight_bits': 8,
    # activation quantize bit num, default is 8
    'activation_bits': 8,
    # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
    'dtype': 'int8',
    # window size for 'range_abs_max' quantization. default is 10000
    'window_size': 10000,
    # The decay coefficient of moving average, default is 0.9
    'moving_rate': 0.9,
    # for dygraph quantization, layers of type in quantizable_layer_type will be quantized
    'quantizable_layer_type': ['Conv2D', 'Linear'],
}


def skip_quant(model):
    """
    If the model has backbone and head, we skip quantizing the conv2d and linear ops
    that belongs the head.
    """
    if not hasattr(model, 'backbone'):
        logger.info("Quantize all target ops")
        return

    logger.info("Quantize all target ops in backbone")
    for name, cur_layer in model.named_sublayers():
        if isinstance(cur_layer, (paddle.nn.Conv2D, paddle.nn.Linear)) \
            and "backbone" not in name:
            cur_layer.skip_quant = True


def main(args):
    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)

    cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size,
        save_interval=args.save_interval)

    train_data_ratio = cfg.train_data_ratio
    val_roots = cfg.val_roots
    dataset_weights = cfg.dataset_weights
    class_weights = cfg.class_weights
    train_transforms = cfg.train_transforms
    val_transforms = cfg.val_transforms
    losses = cfg.loss

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    train_datasets = []
    for _, train_root in enumerate(cfg.train_roots):
        train_datasets.append(
            Dataset(
                train_transforms,
                train_root,
                2,
                train_path=os.path.join(train_root, "train.txt")))
    if len(train_datasets) == 0:
        raise ValueError(
            'The length of train_datasets is 0. Please check if your dataset is valid'
        )
    mixed_train_dataset = MixedDataset(
        train_datasets,
        train_data_ratio,
        transforms=train_transforms,
        num_classes=2)

    if args.do_eval:
        val_datasets = []
        for root in val_roots:
            dataset = Dataset(
                val_transforms,
                root,
                2,
                mode='val',
                val_path=os.path.join(root, 'val.txt'))
            val_datasets.append(dataset)
    else:
        val_datasets = None

    if place == 'gpu' and paddle.distributed.ParallelEnv().nranks > 1:
        # convert bn to sync_bn
        cfg._model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(cfg.model)

    model = cfg.model
    assert args.model_path is not None, "Please set model_path"
    utils.load_entire_model(model, args.model_path)
    logger.info('Loaded trained params of model successfully')

    skip_quant(model)
    quantizer = QAT(config=quant_config)
    quantizer.quantize(model)
    logger.info('Quantize the model successfully')

    train(
        cfg.model,
        mixed_train_dataset,
        val_datasets=val_datasets,
        class_weights=class_weights,
        dataset_weights=dataset_weights,
        optimizer=cfg.optimizer,
        save_dir=args.save_dir,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        resume_model=args.resume_model,
        save_interval=cfg.save_interval,
        log_iters=args.log_iters,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        losses=losses,
        keep_checkpoint_max=args.keep_checkpoint_max)


if __name__ == '__main__':
    args = parse_args()
    main(args)

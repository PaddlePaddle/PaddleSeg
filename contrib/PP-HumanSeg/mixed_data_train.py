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
from paddleseg.utils import get_sys_env, logger, config_check
from paddleseg.transforms import *
from paddleseg.models import *
from paddleseg.models.losses import *
from datasets.humanseg import HumanSeg
from paddleseg.datasets import Dataset
from datasets.mixed_dataset import MixedDataset
from scripts.mixed_data_train import train


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
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
        '--train_roots', dest='train_roots', nargs='+', type=str)
    parser.add_argument(
        '--train_file_lists', dest='train_file_lists', nargs='+', type=str)
    parser.add_argument(
        '--train_data_ratio', dest='train_data_ratio', nargs='+', type=float)
    parser.add_argument('--val_roots', dest='val_roots', nargs='+', type=str)
    parser.add_argument(
        '--dataset_weights', dest='dataset_weights', nargs='+', type=float)
    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=1000)
    parser.add_argument(
        '--model_name', dest='model_name', type=str, default=None)
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

    return parser.parse_args()


def main(args):
    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)

    #========================================= config ======================================
    iters = 10000  # 2gpu 15epoch
    save_interval = 400
    batch_size = 128
    learning_rate = 0.01
    # pretrained = None
    # pretrained = 'saved_model/shufflenetv2_humanseg_192x192/best_model/model.pdparams'

    train_roots = args.train_roots
    train_file_lists = args.train_file_lists
    train_data_ratio = args.train_data_ratio
    val_roots = args.val_roots
    dataset_weights = args.dataset_weights
    class_weights = [0.3, 0.7]

    train_transforms = [
        PaddingByAspectRatio(1.77777778),
        Resize(target_size=[398, 224]),
        ResizeStepScaling(scale_step_size=0),
        RandomRotation(),
        RandomPaddingCrop(crop_size=[398, 224]),
        RandomHorizontalFlip(),
        RandomDistort(),
        RandomBlur(prob=0.3),
        Normalize()
    ]

    val_transforms = [
        PaddingByAspectRatio(1.77777778),
        Resize(target_size=[398, 224]),
        Normalize()
    ]
    #========================================================================================
    train_datasets = []
    for i, train_root in enumerate(train_roots):
        train_datasets.append(
            Dataset(
                train_transforms,
                train_root,
                2,
                train_path=os.path.join(train_root, train_file_lists[i])))
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
        val_dataset0 = Dataset(
            val_transforms,
            val_roots[0],
            2,
            mode='val',
            val_path=os.path.join(val_roots[0], 'val.txt'))
        val_dataset1 = Dataset(
            val_transforms,
            val_roots[1],
            2,
            mode='val',
            val_path=os.path.join(val_roots[1], 'val.txt'))
        val_dataset2 = Dataset(
            val_transforms,
            val_roots[2],
            2,
            mode='val',
            val_path=os.path.join(val_roots[2], 'val.txt'))
        val_datasets = [val_dataset0, val_dataset1, val_dataset2]
        # val_datasets = val_dataset0
    else:
        val_datasets = None

    from paddleseg.models.stdcseg import STDCSeg
    from paddleseg.models.backbones import STDC1, STDC2
    backbone = STDC2(
        pretrained='https://bj.bcebos.com/paddleseg/dygraph/STDCNet2.tar.gz')
    model_class = manager.MODELS[args.model_name]
    model = model_class(2, backbone)
    losses = {
        'types': [
            OhemCrossEntropyLoss(),
            OhemCrossEntropyLoss(),
            OhemCrossEntropyLoss(),
            DetailAggregateLoss()
        ],
        'coef': [1, 1, 1, 1]
    }

    # from paddleseg.models.backbones.lcnet import PPLCNet_x1_0
    # backbone = PPLCNet_x1_0()
    # model_class = manager.MODELS[args.model_name]
    # model = model_class(2, backbone)

    # losses = {
    #     'types': [
    #         MixedLoss(
    #             losses=[CrossEntropyLoss(),
    #                     LovaszSoftmaxLoss()],
    #             coef=[0.8, 0.2])
    #     ],
    #     'coef': [1]
    # }

    lr = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=learning_rate, end_lr=0, power=0.9, decay_steps=iters)
    optimizer = paddle.optimizer.Momentum(
        lr, parameters=model.parameters(), momentum=0.9, weight_decay=0.0005)

    logger.info('iters {}'.format(iters))
    logger.info('save_interval {}'.format(save_interval))
    logger.info('batch_size {}'.format(batch_size))
    logger.info('learning_rate {}'.format(learning_rate))
    # logger.info('pretrained {}'.format(pretrained))
    logger.info('train_roots {}'.format(str(train_roots)))
    logger.info('val_roots {}'.format(str(val_roots)))
    logger.info('dataset_weights {}'.format(str(dataset_weights)))
    logger.info('class_weights {}'.format(str(class_weights)))
    logger.info('train_data_ratio {}'.format(str(train_data_ratio)))

    train(
        model,
        mixed_train_dataset,
        val_datasets=val_datasets,
        class_weights=class_weights,
        dataset_weights=dataset_weights,
        optimizer=optimizer,
        save_dir=args.save_dir,
        iters=iters,
        batch_size=batch_size,
        resume_model=args.resume_model,
        save_interval=save_interval,
        log_iters=args.log_iters,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        losses=losses,
        keep_checkpoint_max=args.keep_checkpoint_max)


if __name__ == '__main__':
    args = parse_args()
    main(args)

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

import paddle

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, config_check
from paddleseg.transforms import *
from paddleseg.models import *
from paddleseg.models.losses import *
from datasets.humanseg import HumanSeg
from datasets.multi_dataset import MultiDataset
from scripts.train import train


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
        '--save_interval',
        dest='save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=1000)
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

    iters = 10000
    save_interval = 400
    batch_size = 128
    learning_rate = 0.1
    pretrained = 'saved_model/shufflenetv2_humanseg_192x192/best_model/model.pdparams'

    dataset_root_list = [
        "data/portrait2600",
        "data/matting_human_half",
        "/ssd2/chulutao/humanseg",
    ]
    valset_weight = [0.2, 0.2, 0.6]
    valset_class_weight = [0.3, 0.7]
    data_ratio = [10, 1, 1]

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

    train_dataset = MultiDataset(
        train_transforms,
        dataset_root_list,
        valset_weight,
        valset_class_weight,
        data_ratio=data_ratio,
        mode='train',
        num_classes=2)
    if train_dataset is None:
        raise RuntimeError(
            'The training dataset is not specified in the configuration file.')
    elif len(train_dataset) == 0:
        raise ValueError(
            'The length of train_dataset is 0. Please check if your dataset is valid'
        )

    if args.do_eval:
        val_dataset0 = MultiDataset(
            val_transforms,
            dataset_root_list,
            valset_weight,
            valset_class_weight,
            data_ratio=data_ratio,
            mode='val',
            num_classes=2,
            val_test_set_rank=0)
        val_dataset1 = MultiDataset(
            val_transforms,
            dataset_root_list,
            valset_weight,
            valset_class_weight,
            data_ratio=data_ratio,
            mode='val',
            num_classes=2,
            val_test_set_rank=1)
        val_dataset2 = MultiDataset(
            val_transforms,
            dataset_root_list,
            valset_weight,
            valset_class_weight,
            data_ratio=data_ratio,
            mode='val',
            num_classes=2,
            val_test_set_rank=2)
        val_dataset = [val_dataset0, val_dataset1, val_dataset2]
    else:
        val_dataset = None

    model = ShuffleNetV2(
        align_corners=False, num_classes=2, pretrained=pretrained)
    losses = {
        'types': [
            MixedLoss(
                losses=[CrossEntropyLoss(),
                        LovaszSoftmaxLoss()],
                coef=[0.8, 0.2])
        ],
        'coef': [1]
    }
    lr = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=learning_rate, end_lr=0, power=0.9, decay_steps=iters)
    optimizer = paddle.optimizer.Momentum(
        lr, parameters=model.parameters(), momentum=0.9, weight_decay=0.0005)

    logger.info('iters {}'.format(iters))
    logger.info('save_interval {}'.format(save_interval))
    logger.info('batch_size {}'.format(batch_size))
    logger.info('learning_rate {}'.format(learning_rate))
    logger.info('pretrained {}'.format(pretrained))
    logger.info('dataset_root_list {}'.format(str(dataset_root_list)))
    logger.info('valset_weight {}'.format(str(valset_weight)))
    logger.info('valset_class_weight {}'.format(str(valset_class_weight)))
    logger.info('data_ratio {}'.format(str(data_ratio)))

    train(
        model,
        train_dataset,
        val_dataset=val_dataset,
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

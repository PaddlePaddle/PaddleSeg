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

from core import train
from model import *
from dataset import HumanDataset
import transforms as T


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    # parser.add_argument(
    #     "--config", dest="cfg", help="The config file.", default=None, type=str)
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
    parser.add_argument(
        '--stage',
        dest='stage',
        help='training stage: 0(simple loss), 1, 2, 3(whole net)',
        type=int,
        required=True,
        choices=[0, 1, 2, 3])
    parser.add_argument(
        '--pretrained_model',
        dest='pretrained_model',
        help='the pretrained model',
        type=str)

    return parser.parse_args()


def main(args):
    paddle.set_device('gpu')

    # 一些模块的组建
    # train_dataset
    # 简单的建立一个数据读取器
    # train_dataset = Dataset()
    t = [
        T.LoadImages(),
        T.RandomCropByAlpha(crop_size=((320, 320), (480, 480), (640, 640))),
        T.Resize(target_size=(320, 320)),
        T.Normalize()
    ]

    train_dataset = HumanDataset(
        dataset_root='/mnt/chenguowei01/datasets/matting/human_matting/',
        transforms=t,
        mode='train')

    # loss
    losses = {'types': [], 'coef': []}
    # encoder-decoder alpha loss
    losses['types'].append(MRSD())
    losses['coef'].append(0.5)
    # compositionnal loss
    losses['types'].append(MRSD())
    losses['coef'].append(0.5)
    # refine alpha loss
    losses['types'].append(MRSD())
    losses['coef'].append(1)

    # model
    # vgg16预训练模型地址： 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VGG16_pretrained.pdparams')
    backbone = VGG16(input_channels=4, pretrained='./VGG16_pretrained.pdparams')
    model = DIM(
        backbone=backbone, stage=args.stage, pretrained=args.pretrained_model)
    print(model.parameters())

    # optimizer
    # 简单的先构建一个优化器
    # lr = paddle.optimizer.lr.PolynomialDecay(
    #     0.001, decay_steps=200000, end_lr=0.0, power=0.9)
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.learning_rate, parameters=model.parameters())

    # 调用train函数进行训练
    train(
        model=model,
        train_dataset=train_dataset,
        optimizer=optimizer,
        losses=losses,
        iters=args.iters,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        save_interval=args.save_interval,
        resume_model=args.resume_model,
        stage=args.stage,
        save_dir=args.save_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)

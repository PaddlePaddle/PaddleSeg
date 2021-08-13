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
import os
from collections import defaultdict

import paddle
import paddle.nn as nn
import paddleseg

from core import train
from model import *
from dataset import HumanMattingDataset
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
        '--pretrained_model',
        dest='pretrained_model',
        help='the pretrained model',
        type=str)
    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='the dataset root directory',
        type=str)
    parser.add_argument(
        '--save_begin_iters',
        dest='save_begin_iters',
        help='The iters saving begin',
        default=None,
        type=int)
    parser.add_argument(
        '--backbone',
        dest='backbone',
        help='The backbone of model. It is one of (MobileNetV2)',
        required=True,
        type=str)
    parser.add_argument(
        '--train_file',
        dest='train_file',
        nargs='+',
        help='Image list for traiing',
        type=str,
        default='train.txt')
    parser.add_argument(
        '--val_file',
        dest='val_file',
        nargs='+',
        help='Image list for evaluation',
        type=str,
        default='val.txt')

    return parser.parse_args()


def main(args):
    paddle.set_device('gpu')

    # 一些模块的组建
    # train_dataset
    # 简单的建立一个数据读取器
    # train_dataset = Dataset()
    t = [
        T.LoadImages(),
        T.RandomCropByAlpha(crop_size=((512, 512), (640, 640), (800, 800))),
        T.Resize(target_size=(512, 512)),
        T.Normalize()
    ]

    train_dataset = HumanMattingDataset(
        dataset_root=args.dataset_root,
        transforms=t,
        mode='train',
        train_file=args.train_file)
    if args.do_eval:
        t = [T.LoadImages(), T.ResizeToIntMult(mult_int=32), T.Normalize()]
        val_dataset = HumanMattingDataset(
            dataset_root=args.dataset_root,
            transforms=t,
            mode='val',
            val_file=args.val_file,
            get_trimap=False)
    else:
        val_dataset = None

    # loss
    losses = defaultdict(list)
    losses['semantic'].append(paddleseg.models.MSELoss())
    losses['detail'].append(paddleseg.models.L1Loss())
    losses['fusion'].append(paddleseg.models.L1Loss())
    losses['fusion'].append(paddleseg.models.L1Loss())

    # model
    #bulid backbone
    pretrained_model = './pretrained_models/' + args.backbone + '_pretrained.pdparams'
    if not os.path.exists(pretrained_model):
        pretrained_model = None
    backbone = eval(args.backbone)(
        input_channels=3, pretrained=pretrained_model)

    model = MODNet(backbone=backbone, pretrained=args.pretrained_model)

    # optimizer
    # 简单的先构建一个优化器
    # lr = paddle.optimizer.lr.PolynomialDecay(
    #     0.001, decay_steps=200000, end_lr=0.0, power=0.9)
    # use adam
    #     optimizer = paddle.optimizer.Adam(
    #         learning_rate=args.learning_rate, parameters=model.parameters())

    #     lr = paddle.optimizer.lr.StepDecay(args.learning_rate, step_size=1000, gamma=0.1, last_epoch=-1, verbose=False)
    boundaries = [20000, 500000, 80000]
    values = [
        args.learning_rate * 0.1**scale for scale in range(len(boundaries) + 1)
    ]
    lr = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=boundaries, values=values, last_epoch=-1, verbose=False)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr,
        momentum=0.9,
        parameters=model.parameters(),
        weight_decay=4e-5)

    # 调用train函数进行训练
    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        losses=losses,
        iters=args.iters,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        save_interval=args.save_interval,
        log_iters=args.log_iters,
        resume_model=args.resume_model,
        save_dir=args.save_dir,
        save_begin_iters=args.save_begin_iters)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)

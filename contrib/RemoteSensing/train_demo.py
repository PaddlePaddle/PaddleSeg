# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import os.path as osp
import argparse
import transforms.transforms as T
from readers.reader import Reader
from models import UNet


def parse_args():
    parser = argparse.ArgumentParser(description='RemoteSensing training')
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='dataset directory',
        default=None,
        type=str)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='model save directory',
        default=None,
        type=str)
    parser.add_argument(
        '--channel',
        dest='channel',
        help='number of data channel',
        default=3,
        type=int)
    parser.add_argument(
        '--num_epochs',
        dest='num_epochs',
        help='number of traing epochs',
        default=100,
        type=int)
    parser.add_argument(
        '--train_batch_size',
        dest='train_batch_size',
        help='training batch size',
        default=4,
        type=int)
    parser.add_argument(
        '--lr', dest='lr', help='learning rate', default=0.01, type=float)
    return parser.parse_args()


args = parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
channel = args.channel
num_epochs = args.num_epochs
train_batch_size = args.train_batch_size
lr = args.lr

# 定义训练和验证时的transforms
train_transforms = T.Compose([
    T.RandomVerticalFlip(0.5),
    T.RandomHorizontalFlip(0.5),
    T.ResizeStepScaling(0.5, 2.0, 0.25),
    T.RandomPaddingCrop(256),
    T.Normalize(mean=[0.5] * channel, std=[0.5] * channel),
])

eval_transforms = T.Compose([
    T.Normalize(mean=[0.5] * channel, std=[0.5] * channel),
])

train_list = osp.join(data_dir, 'train.txt')
val_list = osp.join(data_dir, 'val.txt')
label_list = osp.join(data_dir, 'labels.txt')

# 定义数据读取器
train_reader = Reader(
    data_dir=data_dir,
    file_list=train_list,
    label_list=label_list,
    transforms=train_transforms,
    num_workers=8,
    buffer_size=16,
    shuffle=True,
    parallel_method='thread')

eval_reader = Reader(
    data_dir=data_dir,
    file_list=val_list,
    label_list=label_list,
    transforms=eval_transforms,
    num_workers=8,
    buffer_size=16,
    shuffle=False,
    parallel_method='thread')

model = UNet(
    num_classes=2, input_channel=channel, use_bce_loss=True, use_dice_loss=True)

model.train(
    num_epochs=num_epochs,
    train_reader=train_reader,
    train_batch_size=train_batch_size,
    eval_reader=eval_reader,
    save_interval_epochs=5,
    log_interval_steps=10,
    save_dir=save_dir,
    pretrain_weights=None,
    optimizer=None,
    learning_rate=lr,
    use_vdl=True)

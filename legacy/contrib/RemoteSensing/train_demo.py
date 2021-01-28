# coding: utf8
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

import os.path as osp
import argparse

import transforms.transforms as T
from readers.reader import Reader
from models import UNet, HRNet
from utils import paddle_utils


def parse_args():
    parser = argparse.ArgumentParser(description='RemoteSensing training')
    parser.add_argument(
        '--model_type',
        dest='model_type',
        help="Model type for traing, which is one of ('unet', 'hrnet')",
        type=str,
        default='hrnet')
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
        '--num_classes',
        dest='num_classes',
        help='Number of classes',
        default=None,
        type=int)
    parser.add_argument(
        '--channel',
        dest='channel',
        help='number of data channel',
        default=3,
        type=int)
    parser.add_argument(
        '--clip_min_value',
        dest='clip_min_value',
        help='Min values for clipping data',
        nargs='+',
        default=None,
        type=int)
    parser.add_argument(
        '--clip_max_value',
        dest='clip_max_value',
        help='Max values for clipping data',
        nargs='+',
        default=None,
        type=int)
    parser.add_argument(
        '--mean',
        dest='mean',
        help='Data means',
        nargs='+',
        default=None,
        type=float)
    parser.add_argument(
        '--std',
        dest='std',
        help='Data standard deviation',
        nargs='+',
        default=None,
        type=float)
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


paddle_utils.enable_static()
args = parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
num_classes = args.num_classes
channel = args.channel
clip_min_value = args.clip_min_value
clip_max_value = args.clip_max_value
mean = args.mean
std = args.std
num_epochs = args.num_epochs
train_batch_size = args.train_batch_size
lr = args.lr

# 定义训练和验证时的transforms
train_transforms = T.Compose([
    T.RandomVerticalFlip(0.5),
    T.RandomHorizontalFlip(0.5),
    T.ResizeStepScaling(0.5, 2.0, 0.25),
    T.RandomPaddingCrop(1000),
    T.Clip(min_val=clip_min_value, max_val=clip_max_value),
    T.Normalize(
        min_val=clip_min_value, max_val=clip_max_value, mean=mean, std=std),
])

eval_transforms = T.Compose([
    T.Clip(min_val=clip_min_value, max_val=clip_max_value),
    T.Normalize(
        min_val=clip_min_value, max_val=clip_max_value, mean=mean, std=std),
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
    shuffle=True)

eval_reader = Reader(
    data_dir=data_dir,
    file_list=val_list,
    label_list=label_list,
    transforms=eval_transforms)

if args.model_type == 'unet':
    model = UNet(num_classes=num_classes, input_channel=channel)
elif args.model_type == 'hrnet':
    model = HRNet(num_classes=num_classes, input_channel=channel)
else:
    raise ValueError(
        "--model_type: {} is set wrong, it shold be one of ('unet', "
        "'hrnet')".format(args.model_type))

model.train(
    num_epochs=num_epochs,
    train_reader=train_reader,
    train_batch_size=train_batch_size,
    eval_reader=eval_reader,
    eval_best_metric='miou',
    save_interval_epochs=5,
    log_interval_steps=10,
    save_dir=save_dir,
    learning_rate=lr,
    use_vdl=True)

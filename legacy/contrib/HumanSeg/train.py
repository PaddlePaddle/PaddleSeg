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

import argparse
from datasets.dataset import Dataset
from models import HumanSegMobile, HumanSegLite, HumanSegServer
import transforms

MODEL_TYPE = ['HumanSegMobile', 'HumanSegLite', 'HumanSegServer']


def parse_args():
    parser = argparse.ArgumentParser(description='HumanSeg training')
    parser.add_argument(
        '--model_type',
        dest='model_type',
        help=
        "Model type for traing, which is one of ('HumanSegMobile', 'HumanSegLite', 'HumanSegServer')",
        type=str,
        default='HumanSegMobile')
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='The root directory of dataset',
        type=str)
    parser.add_argument(
        '--train_list',
        dest='train_list',
        help='Train list file of dataset',
        type=str)
    parser.add_argument(
        '--val_list',
        dest='val_list',
        help='Val list file of dataset',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')
    parser.add_argument(
        '--num_classes',
        dest='num_classes',
        help='Number of classes',
        type=int,
        default=2)
    parser.add_argument(
        "--image_shape",
        dest="image_shape",
        help="The image shape for net inputs.",
        nargs=2,
        default=[192, 192],
        type=int)
    parser.add_argument(
        '--num_epochs',
        dest='num_epochs',
        help='Number epochs for training',
        type=int,
        default=100)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size',
        type=int,
        default=128)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=0.01)
    parser.add_argument(
        '--pretrained_weights',
        dest='pretrained_weights',
        help='The path of pretrianed weight',
        type=str,
        default=None)
    parser.add_argument(
        '--resume_weights',
        dest='resume_weights',
        help='The path of resume weight',
        type=str,
        default=None)
    parser.add_argument(
        '--use_vdl',
        dest='use_vdl',
        help='Whether to use visualdl',
        action='store_true')
    parser.add_argument(
        '--save_interval_epochs',
        dest='save_interval_epochs',
        help='The interval epochs for save a model snapshot',
        type=int,
        default=5)

    return parser.parse_args()


def train(args):
    train_transforms = transforms.Compose([
        transforms.Resize(args.image_shape),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize()
    ])

    eval_transforms = transforms.Compose(
        [transforms.Resize(args.image_shape),
         transforms.Normalize()])

    train_dataset = Dataset(
        data_dir=args.data_dir,
        file_list=args.train_list,
        transforms=train_transforms,
        num_workers='auto',
        buffer_size=100,
        parallel_method='thread',
        shuffle=True)

    eval_dataset = None
    if args.val_list is not None:
        eval_dataset = Dataset(
            data_dir=args.data_dir,
            file_list=args.val_list,
            transforms=eval_transforms,
            num_workers='auto',
            buffer_size=100,
            parallel_method='thread',
            shuffle=False)

    if args.model_type == 'HumanSegMobile':
        model = HumanSegMobile(num_classes=2)
    elif args.model_type == 'HumanSegLite':
        model = HumanSegLite(num_classes=2)
    elif args.model_type == 'HumanSegServer':
        model = HumanSegServer(num_classes=2)
    else:
        raise ValueError(
            "--model_type: {} is set wrong, it shold be one of ('HumanSegMobile', "
            "'HumanSegLite', 'HumanSegServer')".format(args.model_type))
    model.train(
        num_epochs=args.num_epochs,
        train_dataset=train_dataset,
        train_batch_size=args.batch_size,
        eval_dataset=eval_dataset,
        save_interval_epochs=args.save_interval_epochs,
        save_dir=args.save_dir,
        pretrained_weights=args.pretrained_weights,
        resume_weights=args.resume_weights,
        learning_rate=args.learning_rate,
        use_vdl=args.use_vdl)


if __name__ == '__main__':
    args = parse_args()
    train(args)

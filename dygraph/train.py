# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import os.path as osp

from paddle.fluid.dygraph.base import to_variable
import numpy as np
import paddle.fluid as fluid

from datasets.dataset import Dataset
import transforms as T
import models
import utils.logging as logging
from utils import get_environ_info
from val import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # params of model
    parser.add_argument(
        '--model_name',
        dest='model_name',
        help="Model type for traing, which is one of ('UNet')",
        type=str,
        default='UNet')

    # params of dataset
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
        '--num_classes',
        dest='num_classes',
        help='Number of classes',
        type=int,
        default=2)
    parser.add_argument(
        '--ingore_index',
        dest='ignore_index',
        help=
        'The pixel equaling ignore_index will not be computed during training',
        type=int,
        default=255)

    # params of training
    parser.add_argument(
        "--input_size",
        dest="input_size",
        help="The image size for net inputs.",
        nargs=2,
        default=[512, 512],
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
        default=2)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=0.01)
    parser.add_argument(
        '--pretrained_model',
        dest='pretrained_model',
        help='The path of pretrianed weight',
        type=str,
        default=None)
    parser.add_argument(
        '--save_interval_epochs',
        dest='save_interval_epochs',
        help='The interval epochs for save a model snapshot',
        type=int,
        default=5)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')

    return parser.parse_args()


def train(model,
          train_dataset,
          eval_dataset=None,
          optimizer=None,
          save_dir='output',
          num_epochs=100,
          batch_size=2,
          pretrained_model=None,
          save_interval_epochs=1,
          num_classes=None):
    if not osp.isdir(save_dir):
        if osp.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir)

    data_generator = train_dataset.generator(
        batch_size=batch_size, drop_last=True)
    num_steps_each_epoch = train_dataset.num_samples // args.batch_size

    for epoch in range(num_epochs):
        for step, data in enumerate(data_generator()):
            images = np.array([d[0] for d in data])
            labels = np.array([d[1] for d in data]).astype('int64')
            images = to_variable(images)
            labels = to_variable(labels)
            loss = model(images, labels, mode='train')
            loss.backward()
            optimizer.minimize(loss)
            logging.info("[TRAIN] Epoch={}/{}, Step={}/{}, loss={}".format(
                epoch + 1, num_epochs, step + 1, num_steps_each_epoch,
                loss.numpy()))

        if (
                epoch + 1
        ) % save_interval_epochs == 0 or num_steps_each_epoch == num_epochs - 1:
            current_save_dir = osp.join(save_dir, "epoch_{}".format(epoch + 1))
            if not osp.isdir(current_save_dir):
                os.makedirs(current_save_dir)
            fluid.save_dygraph(model.state_dict(),
                               osp.join(current_save_dir, 'model'))

            if eval_dataset is not None:
                model.eval()
                evaluate(
                    model,
                    eval_dataset,
                    model_dir=current_save_dir,
                    num_classes=num_classes,
                    batch_size=batch_size,
                    ignore_index=model.ignore_index,
                    epoch_id=epoch + 1)
                model.train()


def arrange_transform(transforms, mode='train'):
    arrange_transform = T.ArrangeSegmenter
    if type(transforms.transforms[-1]).__name__.startswith('Arrange'):
        transforms.transforms[-1] = arrange_transform(mode=mode)
    else:
        transforms.transforms.append(arrange_transform(mode=mode))


def main(args):
    # Creat dataset reader
    train_transforms = T.Compose(
        [T.Resize(args.input_size),
         T.RandomHorizontalFlip(),
         T.Normalize()])
    arrange_transform(train_transforms, mode='train')
    train_dataset = Dataset(
        data_dir=args.data_dir,
        file_list=args.train_list,
        transforms=train_transforms,
        num_workers='auto',
        buffer_size=100,
        parallel_method='thread',
        shuffle=True)
    if args.val_list is not None:
        eval_transforms = T.Compose([T.Resize(args.input_size), T.Normalize()])
        arrange_transform(eval_transforms, mode='eval')
        eval_dataset = Dataset(
            data_dir=args.data_dir,
            file_list=args.val_list,
            transforms=eval_transforms,
            num_workers='auto',
            buffer_size=100,
            parallel_method='thread',
            shuffle=False)

    if args.model_name == 'UNet':
        model = models.UNet(
            num_classes=args.num_classes, ignore_index=args.ignore_index)

    # Creat optimizer
    num_steps_each_epoch = train_dataset.num_samples // args.batch_size
    decay_step = args.num_epochs * num_steps_each_epoch
    lr_decay = fluid.layers.polynomial_decay(
        args.learning_rate, decay_step, end_learning_rate=0, power=0.9)
    optimizer = fluid.optimizer.Momentum(
        lr_decay,
        momentum=0.9,
        parameter_list=model.parameters(),
        regularization=fluid.regularizer.L2Decay(regularization_coeff=4e-5))

    train(
        model,
        train_dataset,
        eval_dataset,
        optimizer,
        save_dir=args.save_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        pretrained_model=args.pretrained_model,
        save_interval_epochs=args.save_interval_epochs,
        num_classes=args.num_classes)


if __name__ == '__main__':
    args = parse_args()
    env_info = get_environ_info()
    if env_info['place'] == 'cpu':
        places = fluid.CPUPlace()
    else:
        places = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(places):
        main(args)

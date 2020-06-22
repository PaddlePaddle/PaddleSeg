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

import paddle.fluid as fluid
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.fluid.io import DataLoader
from paddle.incubate.hapi.distributed import DistributedBatchSampler

from datasets import OpticDiscSeg, Cityscapes
import transforms as T
import models
import utils.logging as logging
from utils import get_environ_info
from utils import load_pretrained_model
from utils import resume
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
        '--dataset',
        dest='dataset',
        help=
        "The dataset you want to train, which is one of ('OpticDiscSeg', 'Cityscapes')",
        type=str,
        default='OpticDiscSeg')

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
        help='Mini batch size of one gpu or cpu',
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
        help='The path of pretrained weight',
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

    return parser.parse_args()


def train(model,
          train_dataset,
          places=None,
          eval_dataset=None,
          optimizer=None,
          save_dir='output',
          num_epochs=100,
          batch_size=2,
          pretrained_model=None,
          resume_model=None,
          save_interval_epochs=1,
          num_classes=None,
          num_workers=8):
    ignore_index = model.ignore_index
    nranks = ParallelEnv().nranks

    start_epoch = 0
    if resume_model is not None:
        start_epoch = resume(optimizer, resume_model)
    elif pretrained_model is not None:
        load_pretrained_model(model, pretrained_model)

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir)

    if nranks > 1:
        strategy = fluid.dygraph.prepare_context()
        model_parallel = fluid.dygraph.DataParallel(model, strategy)

    batch_sampler = DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        places=places,
        num_workers=num_workers,
        return_list=True,
    )

    for epoch in range(start_epoch, num_epochs):
        for step, data in enumerate(loader):
            images = data[0]
            labels = data[1].astype('int64')
            if nranks > 1:
                loss = model_parallel(images, labels, mode='train')
                loss = model_parallel.scale_loss(loss)
                loss.backward()
                model_parallel.apply_collective_grads()
            else:
                loss = model(images, labels, mode='train')
                loss.backward()
            optimizer.minimize(loss)
            model.clear_gradients()
            lr = optimizer.current_step_lr()
            logging.info(
                "[TRAIN] Epoch={}/{}, Step={}/{}, loss={}, lr={}".format(
                    epoch + 1, num_epochs, step + 1, len(batch_sampler),
                    loss.numpy(), lr))

        if ((epoch + 1) % save_interval_epochs == 0
                or epoch == num_epochs - 1) and ParallelEnv().local_rank == 0:
            current_save_dir = os.path.join(save_dir,
                                            "epoch_{}".format(epoch + 1))
            if not os.path.isdir(current_save_dir):
                os.makedirs(current_save_dir)
            fluid.save_dygraph(model.state_dict(),
                               os.path.join(current_save_dir, 'model'))
            fluid.save_dygraph(optimizer.state_dict(),
                               os.path.join(current_save_dir, 'model'))

            if eval_dataset is not None:
                evaluate(
                    model,
                    eval_dataset,
                    places=places,
                    model_dir=current_save_dir,
                    num_classes=num_classes,
                    batch_size=batch_size,
                    ignore_index=ignore_index,
                    epoch_id=epoch + 1)
                model.train()


def main(args):
    env_info = get_environ_info()
    places = fluid.CUDAPlace(ParallelEnv().dev_id) \
        if env_info['place'] == 'cuda' and fluid.is_compiled_with_cuda() \
        else fluid.CPUPlace()

    if args.dataset.lower() == 'opticdiscseg':
        dataset = OpticDiscSeg
    elif args.dataset.lower() == 'cityscapes':
        dataset = Cityscapes
    else:
        raise Exception(
            "The --dataset set wrong. It should be one of ('OpticDiscSeg', 'Cityscapes')"
        )

    with fluid.dygraph.guard(places):
        # Creat dataset reader
        train_transforms = T.Compose([
            T.Resize(args.input_size),
            T.RandomHorizontalFlip(),
            T.Normalize()
        ])
        train_dataset = dataset(transforms=train_transforms, mode='train')

        eval_dataset = None
        if args.do_eval:
            eval_transforms = T.Compose(
                [T.Resize(args.input_size),
                 T.Normalize()])
            eval_dataset = dataset(transforms=eval_transforms, mode='eval')

        if args.model_name == 'UNet':
            model = models.UNet(
                num_classes=train_dataset.num_classes, ignore_index=255)

        # Creat optimizer
        # todo, may less one than len(loader)
        num_steps_each_epoch = len(train_dataset) // (
            args.batch_size * ParallelEnv().nranks)
        print(num_steps_each_epoch, 'num_steps_each_epoch')
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
            places=places,
            eval_dataset=eval_dataset,
            optimizer=optimizer,
            save_dir=args.save_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            pretrained_model=args.pretrained_model,
            save_interval_epochs=args.save_interval_epochs,
            num_classes=train_dataset.num_classes,
            num_workers=args.num_workers)


if __name__ == '__main__':
    args = parse_args()
    main(args)

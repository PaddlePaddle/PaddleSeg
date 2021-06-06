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
    parser = argparse.ArgumentParser(description="Model training")

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
    model = DIM(backbone=backbone)

    # optimizer
    # 简单的先构建一个优化器
    lr = paddle.optimizer.lr.PolynomialDecay(
        0.001, decay_steps=200000, end_lr=0.0, power=0.9)
    optimizer = paddle.optimizer.Adam(
        learning_rate=lr, parameters=model.parameters())

    # 调用train函数进行训练
    train(
        model=model,
        train_dataset=train_dataset,
        optimizer=optimizer,
        losses=losses,
        iters=100000,
        batch_size=16,
        num_workers=16,
        use_vdl=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)

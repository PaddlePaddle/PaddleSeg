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

from core import evaluate
from model import *
from dataset import HumanDataset
import transforms as T


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    # parser.add_argument(
    #     "--config", dest="cfg", help="The config file.", default=None, type=str)

    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for evaluation',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output/results')
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)
    parser.add_argument(
        '--stage',
        dest='stage',
        help='training stage: 0(simple loss), 1, 2, 3(whole net)',
        type=int,
        required=True,
        choices=[0, 1, 2, 3])

    return parser.parse_args()


def main(args):
    paddle.set_device('gpu')

    # 一些模块的组建
    # train_dataset
    # 简单的建立一个数据读取器
    # train_dataset = Dataset()
    t = [T.LoadImages(), T.Normalize()]

    eval_dataset = HumanDataset(
        dataset_root='/mnt/chenguowei01/datasets/matting/human_matting/',
        transforms=t,
        mode='val')

    # model
    backbone = VGG16(input_channels=4)
    model = DIM(backbone=backbone, stage=args.stage, pretrained=args.model_path)

    # 调用train函数进行训练
    evaluate(
        model=model,
        eval_dataset=eval_dataset,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        save_results=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)

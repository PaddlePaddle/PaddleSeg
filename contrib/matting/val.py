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
from dataset import HumanMattingDataset
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
        '--backbone',
        dest='backbone',
        help='The backbone of model. It is one of (MobileNetV2)',
        required=True,
        type=str)
    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='the dataset root directory',
        type=str)
    parser.add_argument(
        '--val_file',
        dest='val_file',
        nargs='+',
        help='Image list for evaluation',
        type=str,
        default='val.txt')
    parser.add_argument(
        '--save_results',
        dest='save_results',
        help='save prediction alphe while evaluation',
        action='store_true')

    return parser.parse_args()


def main(args):
    paddle.set_device('gpu')
    #     T.ResizeByLong(long_size=1024),
    t = [T.LoadImages(), T.ResizeToIntMult(mult_int=32), T.Normalize()]
    #     t = [T.LoadImages(), T.LimitLong(max_long=2048), T.ResizeToIntMult(mult_int=32), T.Normalize()]

    eval_dataset = HumanMattingDataset(
        dataset_root=args.dataset_root,
        transforms=t,
        mode='val',
        val_file=args.val_file,
        get_trimap=False)

    # model
    backbone = eval(args.backbone)(input_channels=3)

    model = MODNet(backbone=backbone, pretrained=args.model_path)

    # 调用evaluate函数进行训练
    evaluate(
        model=model,
        eval_dataset=eval_dataset,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        save_results=args.save_results)


if __name__ == '__main__':
    args = parse_args()
    main(args)

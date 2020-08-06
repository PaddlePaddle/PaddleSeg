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

from paddle.fluid.dygraph.base import to_variable
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.parallel import ParallelEnv
import cv2
import tqdm

from datasets import DATASETS
import transforms as T
from models import MODELS
import utils
import utils.logging as logging
from utils import get_environ_info
from core import infer


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # params of model
    parser.add_argument(
        '--model_name',
        dest='model_name',
        help='Model type for testing, which is one of {}'.format(
            str(list(MODELS.keys()))),
        type=str,
        default='UNet')

    # params of dataset
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help="The dataset you want to test, which is one of {}".format(
            str(list(DATASETS.keys()))),
        type=str,
        default='OpticDiscSeg')

    # params of prediction
    parser.add_argument(
        "--input_size",
        dest="input_size",
        help="The image size for net inputs.",
        nargs=2,
        default=[512, 512],
        type=int)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size',
        type=int,
        default=2)
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='The path of model for evaluation',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./output/result')

    return parser.parse_args()


def main(args):
    env_info = get_environ_info()
    places = fluid.CUDAPlace(ParallelEnv().dev_id) \
        if env_info['place'] == 'cuda' and fluid.is_compiled_with_cuda() \
        else fluid.CPUPlace()

    if args.dataset not in DATASETS:
        raise Exception('--dataset is invalid. it should be one of {}'.format(
            str(list(DATASETS.keys()))))
    dataset = DATASETS[args.dataset]

    with fluid.dygraph.guard(places):
        test_transforms = T.Compose([T.Resize(args.input_size), T.Normalize()])
        test_dataset = dataset(transforms=test_transforms, mode='test')

        if args.model_name not in MODELS:
            raise Exception(
                '--model_name is invalid. it should be one of {}'.format(
                    str(list(MODELS.keys()))))
        model = MODELS[args.model_name](num_classes=test_dataset.num_classes)

        infer(
            model,
            model_dir=args.model_dir,
            test_dataset=test_dataset,
            save_dir=args.save_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)

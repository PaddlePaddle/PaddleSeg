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

import paddle.fluid as fluid
from paddle.fluid.dygraph.parallel import ParallelEnv

from dygraph.datasets import DATASETS
import dygraph.transforms as T
from dygraph.cvlibs import manager
from dygraph.utils import get_environ_info
from dygraph.core import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # params of model
    parser.add_argument(
        '--model_name',
        dest='model_name',
        help='Model type for evaluation, which is one of {}'.format(
            str(list(manager.MODELS.components_dict.keys()))),
        type=str,
        default='UNet')

    # params of dataset
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help="The dataset you want to evaluation, which is one of {}".format(
            str(list(DATASETS.keys()))),
        type=str,
        default='OpticDiscSeg')
    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help="dataset root directory",
        type=str,
        default=None)

    # params of evaluate
    parser.add_argument(
        "--input_size",
        dest="input_size",
        help="The image size for net inputs.",
        nargs=2,
        default=[512, 512],
        type=int)
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='The path of model for evaluation',
        type=str,
        default=None)

    return parser.parse_args()


def main(args):
    env_info = get_environ_info()
    places = fluid.CUDAPlace(ParallelEnv().dev_id) \
        if env_info['Paddle compiled with cuda'] and env_info['GPUs used'] \
        else fluid.CPUPlace()

    if args.dataset not in DATASETS:
        raise Exception('`--dataset` is invalid. it should be one of {}'.format(
            str(list(DATASETS.keys()))))
    dataset = DATASETS[args.dataset]

    with fluid.dygraph.guard(places):
        eval_transforms = T.Compose([T.Resize(args.input_size), T.Normalize()])
        eval_dataset = dataset(
            dataset_root=args.dataset_root,
            transforms=eval_transforms,
            mode='val')

        model = manager.MODELS[args.model_name](
            num_classes=eval_dataset.num_classes)

        evaluate(
            model,
            eval_dataset,
            model_dir=args.model_dir,
            num_classes=eval_dataset.num_classes)


if __name__ == '__main__':
    args = parse_args()
    main(args)

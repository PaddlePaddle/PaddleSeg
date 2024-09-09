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
import os

import paddle
import yaml

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import logger, utils
from paddleseg.deploy.export import WrappedModel
from paddleseg.core.export import export


def parse_args():
    parser = argparse.ArgumentParser(description='Export Inference Model.')
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help='The path of trained weights for exporting inference model',
        type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the exported inference model',
        type=str,
        default='./output/inference_model')
    parser.add_argument(
        "--input_shape",
        nargs='+',
        help=
        "Export the model with fixed input shape, e.g., `--input_shape 1 3 1024 1024`.",
        type=int,
        default=None)
    parser.add_argument(
        '--output_op',
        choices=['argmax', 'softmax', 'none'],
        default="argmax",
        help=
        "Select the op to be appended to the last of inference model, default: argmax."
        "In PaddleSeg, the output of trained model is logit (H*C*H*W). We can apply argmax and"
        "softmax op to the logit according the actual situation.")
    parser.add_argument('--for_fd',
                        action='store_true',
                        help="Export the model to FD-compatible format.")

    return parser.parse_args()


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    export(args)


if __name__ == '__main__':
    args = parse_args()
    main(args)

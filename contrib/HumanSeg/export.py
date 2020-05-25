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

import models
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Export model')
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='Model path for exporting',
        type=str)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the export model',
        type=str,
        default='./output/export')
    return parser.parse_args()


def export(args):
    model = models.load_model(args.model_dir)
    model.export_inference_model(args.save_dir)


if __name__ == '__main__':
    args = parse_args()
    export(args)

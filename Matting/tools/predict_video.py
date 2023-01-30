# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys

import paddle
import paddleseg
from paddleseg.cvlibs import manager
from paddleseg.utils import get_sys_env, logger

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..'))

manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()

import ppmatting
from ppmatting.core import predict_video
from ppmatting.utils import Config, MatBuilder


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)

    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for prediction',
        type=str,
        default=None)
    parser.add_argument(
        '--video_path',
        dest='video_path',
        help='The path of video',
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output/results')
    parser.add_argument(
        '--fg_estimate',
        default=True,
        type=eval,
        choices=[True, False],
        help='Whether to estimate foreground when predicting.')
    parser.add_argument(
        '--device',
        dest='device',
        help='Set the device type, which may be GPU, CPU or XPU.',
        default='gpu',
        type=str)

    return parser.parse_args()


def main(args):
    assert args.cfg is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(args.cfg)
    builder = MatBuilder(cfg)

    paddleseg.utils.show_env_info()
    paddleseg.utils.show_cfg_info(cfg)
    paddleseg.utils.set_device(args.device)

    model = builder.model
    transforms = ppmatting.transforms.Compose(builder.val_transforms)

    predict_video(
        model,
        model_path=args.model_path,
        transforms=transforms,
        video_path=args.video_path,
        save_dir=args.save_dir,
        fg_estimate=args.fg_estimate)


if __name__ == '__main__':
    args = parse_args()
    main(args)

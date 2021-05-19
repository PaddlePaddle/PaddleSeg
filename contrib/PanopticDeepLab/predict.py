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
from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, config_check

from core import predict
from datasets import CityscapesPanoptic
from models import PanopticDeepLab


def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    # params of prediction
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for prediction',
        type=str,
        default=None)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help=
        'The path of image, it can be a file or a directory including images',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predicted results',
        type=str,
        default='./output/result')
    parser.add_argument(
        '--threshold',
        dest='threshold',
        help='Threshold applied to center heatmap score',
        type=float,
        default=0.1)
    parser.add_argument(
        '--nms_kernel',
        dest='nms_kernel',
        help='NMS max pooling kernel size',
        type=int,
        default=7)
    parser.add_argument(
        '--top_k',
        dest='top_k',
        help='Top k centers to keep',
        type=int,
        default=200)

    return parser.parse_args()


def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be an image file or a directory including images'
        )

    if len(image_list) == 0:
        raise RuntimeError('There are not image file in `--image_path`')

    return image_list, image_dir


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)
    val_dataset = cfg.val_dataset
    if not val_dataset:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    transforms = val_dataset.transforms
    image_list, image_dir = get_image_list(args.image_path)
    logger.info('Number of predict images = {}'.format(len(image_list)))

    config_check(cfg, val_dataset=val_dataset)

    predict(
        model,
        model_path=args.model_path,
        transforms=transforms,
        thing_list=val_dataset.thing_list,
        label_divisor=val_dataset.label_divisor,
        stuff_area=val_dataset.stuff_area,
        ignore_index=val_dataset.ignore_index,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=args.save_dir,
        threshold=args.threshold,
        nms_kernel=args.nms_kernel,
        top_k=args.top_k)


if __name__ == '__main__':
    args = parse_args()
    main(args)

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

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger

from core import predict
from model import *
from dataset import MattingDataset


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
        '--image_path',
        dest='image_path',
        help=
        'The path of image, it can be a file or a directory including images',
        type=str,
        default=None)
    parser.add_argument(
        '--trimap_path',
        dest='trimap_path',
        help=
        'The path of trimap, it can be a file or a directory including images. '
        'The image should be the same as image when it is a directory.',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output/results')

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
        else:
            image_dir = os.path.dirname(image_path)
            with open(image_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line.split()) > 1:
                        raise RuntimeError(
                            'There should be only one image path per line in `image_path` file. Wrong line: {}'
                            .format(line))
                    image_list.append(os.path.join(image_dir, line))
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
        image_list.sort()
    else:
        raise FileNotFoundError(
            '`image_path` is not found. it should be an image file or a directory including images'
        )

    if len(image_list) == 0:
        raise RuntimeError('There are not image file in `image_path`')

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
    if val_dataset is None:
        raise RuntimeError(
            'The verification dataset is not specified in the configuration file.'
        )
    elif len(val_dataset) == 0:
        raise ValueError(
            'The length of val_dataset is 0. Please check if your dataset is valid'
        )

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    transforms = val_dataset.transforms

    image_list, image_dir = get_image_list(args.image_path)
    if args.trimap_path is None:
        trimap_list = None
    else:
        trimap_list, _ = get_image_list(args.trimap_path)
    logger.info('Number of predict images = {}'.format(len(image_list)))

    predict(
        model,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        trimap_list=trimap_list,
        save_dir=args.save_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)

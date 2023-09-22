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

import paddle

from paddleseg.core import predict
from paddleseg.cvlibs import Config, SegBuilder, manager
from paddleseg.transforms import Compose
from paddleseg.utils import get_image_list, get_sys_env, logger, utils


def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    # Common params
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help='The path of trained weights for prediction.',
        type=str)
    parser.add_argument(
        '--image_path',
        help='The image to predict, which can be a path of image, or a file list containing image paths, or a directory including images',
        type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the predicted results.',
        type=str,
        default='./output/result')
    parser.add_argument(
        '--device',
        help='Set the device place for predicting model.',
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)
    parser.add_argument(
        '--device_id',
        help='Set the device id for predicting model.',
        default=0,
        type=int)

    # Data augment params
    parser.add_argument(
        '--aug_pred',
        help='Whether to use mulit-scales and flip augment for prediction',
        action='store_true')
    parser.add_argument(
        '--scales',
        nargs='+',
        help='Scales for augment, e.g., `--scales 0.75 1.0 1.25`.',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        help='Whether to use flip horizontally augment',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        help='Whether to use flip vertically augment',
        action='store_true')

    # Sliding window evaluation params
    parser.add_argument(
        '--is_slide',
        help='Whether to predict images in sliding window method',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        nargs=2,
        help='The crop size of sliding window, the first is width and the second is height.'
        'For example, `--crop_size 512 512`',
        type=int)
    parser.add_argument(
        '--stride',
        nargs=2,
        help='The stride of sliding window, the first is width and the second is height.'
        'For example, `--stride 512 512`',
        type=int)

    # Custom color map
    parser.add_argument(
        '--custom_color',
        nargs='+',
        help='Save images with a custom color map. Default: None, use paddleseg\'s default color map.',
        type=int)

    # Set multi-label mode
    parser.add_argument(
        '--use_multilabel',
        action='store_true',
        default=False,
        help='Whether to enable multilabel mode. Default: False.')

    return parser.parse_args()


def merge_test_config(cfg, args):
    test_config = cfg.test_config
    if 'aug_eval' in test_config:
        test_config.pop('aug_eval')
    if 'auc_roc' in test_config:
        test_config.pop('auc_roc')
    if args.aug_pred:
        test_config['aug_pred'] = args.aug_pred
        test_config['scales'] = args.scales
        test_config['flip_horizontal'] = args.flip_horizontal
        test_config['flip_vertical'] = args.flip_vertical
    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride
    if args.custom_color:
        test_config['custom_color'] = args.custom_color
    if args.use_multilabel:
        test_config['use_multilabel'] = args.use_multilabel
    return test_config


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(args.config)
    builder = SegBuilder(cfg)
    test_config = merge_test_config(cfg, args)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    if args.device != 'cpu':
        device = f"{args.device}:{args.device_id}"
    else:
        device = args.device
    utils.set_device(device)

    model = builder.model
    transforms = Compose(builder.val_transforms)
    image_list, image_dir = get_image_list(args.image_path)
    logger.info('The number of images: {}'.format(len(image_list)))

    predict(
        model,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=args.save_dir,
        **test_config)


if __name__ == '__main__':
    args = parse_args()
    main(args)

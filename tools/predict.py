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

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, get_image_list, utils
from paddleseg.core import predict
from paddleseg.transforms import Compose


def parse_args():
    '''
    Some input params of previous predict.py are moved to config file.
    Please use `-o` or `--opts` to set these params, such as
    aug_eval, scales and so on.
    '''
    parser = argparse.ArgumentParser(description='Model prediction')
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help='The path of model weight for prediction',
        type=str)
    parser.add_argument(
        '--image_path',
        help='mage_path can be the path of a image, a file list containing image paths, or a directory including images',
        type=str)
    parser.add_argument(
        '-o',
        '--opts',
        help='Update the key-value pairs in config file, such as '
        '--o test.aug_eval=True test.scales=0.75,1.0,1.25',
        nargs='+')
    return parser.parse_args()


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config.'
    cfg = Config(args.config, opts=args.opts)

    utils.show_env_info()
    utils.show_cfg_info(cfg)

    utils.set_seed(cfg.global_params('seed'))
    utils.set_device(cfg.global_params('device'))

    transforms = Compose(cfg.val_transforms)
    image_list, image_dir = get_image_list(args.image_path)
    logger.info('The number of images is {}'.format(len(image_list)))

    predict(
        model=cfg.model,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=cfg.global_params('save_dir'),
        aug_pred=cfg.test_params('aug_eval'),
        scales=cfg.test_params('scales'),
        flip_horizontal=cfg.test_params('flip_horizontal'),
        flip_vertical=cfg.test_params('flip_vertical'),
        is_slide=cfg.test_params('is_slide'),
        stride=cfg.test_params('stride'),
        crop_size=cfg.test_params('crop_size'))


if __name__ == '__main__':
    args = parse_args()
    main(args)

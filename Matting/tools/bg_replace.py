# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import cv2
import numpy as np
import paddle
from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..'))

manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()

import ppmatting
from ppmatting.core import predict
from ppmatting.utils import get_image_list, estimate_foreground_ml


def parse_args():
    parser = argparse.ArgumentParser(
        description='PP-HumanSeg inference for video')
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for prediction',
        type=str,
        default=None)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='Image including human',
        type=str,
        default=None)
    parser.add_argument(
        '--trimap_path',
        dest='trimap_path',
        help='The path of trimap',
        type=str,
        default=None)
    parser.add_argument(
        '--background',
        dest='background',
        help='Background for replacing. It is a string which specifies the background color (r,g,b,w) or a path to background image. If not specified, a green background is used.',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./output')
    parser.add_argument(
        '--fg_estimate',
        default=True,
        type=eval,
        choices=[True, False],
        help='Whether to estimate foreground when predicting.')

    return parser.parse_args()


def main(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'
    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(args.cfg)

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    model = cfg.model
    transforms = ppmatting.transforms.Compose(cfg.val_transforms)

    alpha, fg = predict(
        model,
        model_path=args.model_path,
        transforms=transforms,
        image_list=[args.image_path],
        trimap_list=[args.trimap_path],
        save_dir=args.save_dir,
        fg_estimate=args.fg_estimate)

    img_ori = cv2.imread(args.image_path)
    bg = get_bg(args.background, img_ori.shape)
    alpha = alpha / 255.0
    alpha = alpha[:, :, np.newaxis]
    com = alpha * fg + (1 - alpha) * bg
    com = com.astype('uint8')
    com_save_path = os.path.join(args.save_dir,
                                 os.path.basename(args.image_path))
    cv2.imwrite(com_save_path, com)


def get_bg(background, img_shape):
    bg = np.zeros(img_shape)
    if background == 'r':
        bg[:, :, 2] = 255
    elif background is None or background == 'g':
        bg[:, :, 1] = 255
    elif background == 'b':
        bg[:, :, 0] = 255
    elif background == 'w':
        bg[:, :, :] = 255

    elif not os.path.exists(background):
        raise Exception('The --background is not existed: {}'.format(
            background))
    else:
        bg = cv2.imread(background)
        bg = cv2.resize(bg, (img_shape[1], img_shape[0]))
    return bg


if __name__ == "__main__":
    args = parse_args()
    main(args)

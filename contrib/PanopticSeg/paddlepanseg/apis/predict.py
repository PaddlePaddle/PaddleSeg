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
from paddleseg.utils import get_sys_env, logger

from paddlepanseg.core import predict
from paddlepanseg.cvlibs import manager, Config
from paddlepanseg.transforms import constr_test_transforms


def parse_pred_args(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Model prediction")

    # params of prediction
    parser.add_argument(
        '--config', dest='cfg', help="Config file.", default=None, type=str)
    parser.add_argument(
        '--model_path',
        help="Path of the model for prediction.",
        type=str,
        default=None)
    parser.add_argument(
        '--image_path',
        help="Path of the input image, which can be a file or a directory that contains images.",
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        help="Directory to save the predicted results.",
        type=str,
        default="./output/result")

    return parser.parse_args(*args, **kwargs)


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
            "The value of `--image_path` is invalid. It should be an image file or a directory that contains images."
        )

    if len(image_list) == 0:
        raise RuntimeError("No image file is found.")

    return image_list, image_dir


def pred_with_args(args):
    env_info = get_sys_env()
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError("No configuration file has been specified.")

    cfg = Config(args.cfg)
    val_dataset = cfg.val_dataset
    if not val_dataset:
        raise RuntimeError(
            "The validation dataset is not specified in the configuration file.")

    msg = "\n---------------Config Information---------------\n"
    msg += str(cfg)
    msg += "------------------------------------------------"
    logger.info(msg)

    model = cfg.model
    transforms = constr_test_transforms(val_dataset.transforms)
    image_list, image_dir = get_image_list(args.image_path)
    logger.info("Number of images for prediction = {}".format(len(image_list)))

    predict(
        model,
        model_path=args.model_path,
        transforms=transforms,
        postprocessor=cfg.postprocessor,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=args.save_dir)


if __name__ == '__main__':
    args = parse_pred_args()
    pred_with_args(args)

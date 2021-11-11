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
from paddleseg.utils import get_sys_env, logger, config_check
from paddleseg.core import predict
from paddleseg.transforms import *
from paddleseg.models import *

from datasets.humanseg import HumanSeg
from datasets.humanseg import HumanSeg


def parse_args():
    parser = argparse.ArgumentParser(description='Model predict')
    # params of training
    parser.add_argument(
        '--model_name', dest='model_name', type=str, default=None)
    parser.add_argument('--model_dir', dest='model_dir', type=str, default=None)
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
                        line = line.split()[0]
                    print(line)
                    image_list.append(os.path.join(image_dir, line))
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

    transforms = Compose([
        PaddingByAspectRatio(1.77777778),
        Resize(target_size=[398, 224]),
        Normalize()
    ])

    from paddleseg.models.backbones.lcnet import PPLCNet_x1_0
    backbone = PPLCNet_x1_0()
    model_class = manager.MODELS[args.model_name]
    model = model_class(2, backbone)

    save_dir_ = args.model_dir
    model_path = os.path.join(save_dir_, 'best_model/model.pdparams')

    image_path0 = "data/portrait14k/small_test.txt"
    image_list, image_dir = get_image_list(image_path0)
    logger.info('Number of predict images = {}'.format(len(image_list)))
    save_dir = os.path.join(save_dir_, 'predict_portrait14k')
    predict(
        model,
        model_path=model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=save_dir,
    )

    image_path1 = "data/matting_human_half/small_test.txt"
    # image_path1 = "data/matting_human_half/test.txt"
    image_list, image_dir = get_image_list(image_path1)
    logger.info('Number of predict images = {}'.format(len(image_list)))
    save_dir = os.path.join(save_dir_, 'matting_human_half')
    predict(
        model,
        model_path=model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=save_dir,
    )

    image_path2 = "/ssd2/chulutao/humanseg/small_test.txt"
    image_list, image_dir = get_image_list(image_path2)
    logger.info('Number of predict images = {}'.format(len(image_list)))
    save_dir = os.path.join(save_dir_, 'vis')
    predict(
        model,
        model_path=model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=save_dir,
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)

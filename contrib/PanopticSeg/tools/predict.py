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

from paddleseg.utils import logger, get_image_list, utils

from paddlepanseg.core import predict
from paddlepanseg.cvlibs import Config, make_default_builder
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


def pred_with_args(args):
    if not args.cfg:
        raise RuntimeError("No configuration file has been specified.")
    cfg = Config(args.cfg)
    builder = make_default_builder(cfg)

    utils.show_env_info()
    utils.show_cfg_info(cfg)

    model = builder.model
    val_dataset = builder.val_dataset
    transforms = constr_test_transforms(val_dataset.transforms)
    image_list, image_dir = get_image_list(args.image_path)
    logger.info("Number of images for prediction = {}.".format(len(image_list)))

    predict(
        model,
        model_path=args.model_path,
        transforms=transforms,
        postprocessor=builder.postprocessor,
        image_list=image_list,
        label_divisor=val_dataset.label_divisor,
        ignore_index=val_dataset.ignore_index,
        image_dir=image_dir,
        colormap=val_dataset.get_colormap(),
        save_dir=args.save_dir)


if __name__ == '__main__':
    args = parse_pred_args()
    pred_with_args(args)

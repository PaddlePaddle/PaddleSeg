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
    hstr = "Model Prediction. \n\n"\
           "Single-GPU example: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/predict.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        --image_path data/optic_disc_seg/JPEGImages/H0003.jpg \\\n"\
           "        -o global.save_dir=output \n\n"\
           "Multi-GPU example: \n"\
           "    export CUDA_VISIBLE_DEVICES=0,1 \n"\
           "    python -m paddle.distributed.launch tools/predict.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        --image_path data/optic_disc_seg/JPEGImages/ \\\n"\
           "        -o global.save_dir=output/optic_disc_seg \n\n"\
           "Evaluation with data augmentation: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/predict.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        --image_path data/optic_disc_seg/JPEGImages/H0003.jpg \\\n"\
           "        -o global.save_dir=output test.is_aug=True test.scales=0.75,1.0,1.25 test.flip_horizontal=True \n\n"\
           "Evaluation with slide windows method: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/predict.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        --image_path data/optic_disc_seg/JPEGImages/H0003.jpg \\\n"\
           "        -o global.save_dir=output test.is_slide=True test.crop_size=256,256 test.stride=256,256 \n\n"\
           "Use `-o` or `--opts` to update key-value params in config file. Some common params are explained as follows:\n" \
           "    global.device       Set the running device. It should be cpu, gpu, xpu, npu or mlu.\n" \
           "    global.save_dir     Set the directory to save result images.\n" \
           "    test.is_aug         Whether to enable data augmentation. It should be True or False.\n" \
           "    test.scales         Set the resize scales in data augmentation. When test.is_aug=False, test.scales is invalid.\n" \
           "    test.flip_horizontal    Whether to flip horizontal in data augmentation. When test.is_aug=False, test.flip_horizontal is invalid.\n" \
           "    test.flip_vertical      Whether to flip vertical in data augmentation. When test.is_aug=False, test.flip_vertical is invalid.\n" \
           "    test.is_slide       Whether to test image with slide windows method. It should be True or False.\n" \
           "    test.crop_size      Set the crop size in data slide windows testing. When test.is_slide=False, test.crop_size is invalid.\n" \
           "    test.stride         Set the stride in slide windows testing. When test.is_slide=False, test.stride is invalid.\n"
    parser = argparse.ArgumentParser(
        description=hstr, formatter_class=argparse.RawTextHelpFormatter)
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
        help='Update the key-value pairs in config file',
        nargs='+')
    return parser.parse_args()


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config.'
    cfg = Config(args.config, opts=args.opts)

    utils.show_env_info()
    utils.show_cfg_info(cfg)

    utils.set_seed(cfg.global_config('seed'))
    utils.set_device(cfg.global_config('device'))

    transforms = Compose(cfg.val_transforms)
    image_list, image_dir = get_image_list(args.image_path)
    logger.info('The number of images is {}'.format(len(image_list)))

    predict(
        model=cfg.model,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=cfg.global_config('save_dir'),
        aug_pred=cfg.test_config('is_aug'),
        scales=cfg.test_config('scales'),
        flip_horizontal=cfg.test_config('flip_horizontal'),
        flip_vertical=cfg.test_config('flip_vertical'),
        is_slide=cfg.test_config('is_slide'),
        stride=cfg.test_config('stride'),
        crop_size=cfg.test_config('crop_size'))


if __name__ == '__main__':
    args = parse_args()
    main(args)

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

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import logger, get_image_list, utils
from paddleseg.core import predict
from paddleseg.transforms import Compose


def parse_args():
    hstr = "Make prediction with trained models \n\n"\
           "Example 1: Make prediction with a single GPU: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/predict.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        --image_path data/optic_disc_seg/JPEGImages/H0003.jpg \\\n"\
           "        --save_dir=output \n\n"\
           "Example 2: Make prediction with multiple GPUs: \n"\
           "    export CUDA_VISIBLE_DEVICES=0,1 \n"\
           "    python -m paddle.distributed.launch tools/predict.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        --image_path data/optic_disc_seg/JPEGImages/ \\\n"\
           "        --save_dir=output/optic_disc_seg \n\n"\
           "Example 3: Make prediction with test-time data augmentation: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/predict.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        --image_path data/optic_disc_seg/JPEGImages/H0003.jpg \\\n"\
           "        -o test.is_aug=True test.scales=0.75,1.0,1.25 test.flip_horizontal=True \n\n"\
           "Example 4: Make prediction using sliding windows: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/predict.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        --image_path data/optic_disc_seg/JPEGImages/H0003.jpg \\\n"\
           "        -o test.is_slide=True test.crop_size=256,256 test.stride=256,256 \n\n"\
           "Use `-o` or `--opts` to overwrite key-value config items. Some common configurations are explained as follows:\n" \
           "    global.device       Set the running device. It should be 'cpu', 'gpu', 'xpu', 'npu', or 'mlu'.\n" \
           "    global.num_workers  Set the number of workers to read and process images.\n" \
           "    test.is_aug         Whether or not to enable test-time data augmentation. It should be either True or False.\n" \
           "    test.scales         Set the image scaling in test-time data augmentation. Invalidated when `test.is_aug` is False.\n" \
           "    test.flip_horizontal    Whether or not to implement horizontal flip in test-time data augmentation. Invalidated when `test.is_aug` is False.\n" \
           "    test.flip_vertical      Whether or not to implement vertical flip in test-time data augmentation. Invalidated when `test.is_aug` is False.\n" \
           "    test.is_slide       Whether or not to use sliding windows. It should be either True or False.\n" \
           "    test.crop_size      Set the size of sliding windows used for testing. Invalidated when `test.is_slide` is False.\n" \
           "    test.stride         Set the stride of sliding windows used fortesting. Invalidated when `test.is_slide` is False.\n"

    parser = argparse.ArgumentParser(
        description=hstr, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--config', help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help="The path of trained weights to be loaded for prediction.",
        type=str)
    parser.add_argument(
        '--image_path',
        help="The image to predict, which can be a path of image, or a file list containing image paths, or a directory including images.",
        type=str)
    parser.add_argument(
        '--save_dir',
        help="The directory for saving the predicted results.",
        type=str,
        default='./output/result')
    parser.add_argument(
        '--custom_color',
        nargs='+',
        help="Save images with a custom color map. Default: None, using a default color map.",
        type=int)
    parser.add_argument(
        '--opts',
        help="Specify key-value pairs to update configurations.",
        default=None,
        nargs='+')

    return parser.parse_args()


def main(args):
    assert args.config is not None, \
        "No configuration file has been specified. Please set `--config`."
    cfg = Config(args.config)
    global_cfg = cfg.global_cfg
    builder = SegBuilder(cfg)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_device(global_cfg['device'])

    model = builder.model
    transforms = Compose(builder.val_transforms)
    image_list, image_dir = get_image_list(args.image_path)
    logger.info("The number of images: {}".format(len(image_list)))

    test_cfg = cfg.test_cfg
    for unused_key in ('aug_eval', 'auc_roc'):
        if unused_key in test_cfg:
            # Inplace modification
            test_cfg.pop(unused_key)

    predict(
        model,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=args.save_dir,
        custom_color=args.custom_color,
        **test_cfg)


if __name__ == '__main__':
    args = parse_args()
    main(args)

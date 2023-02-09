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
from paddleseg.core import evaluate
from paddleseg.utils import logger, utils


def parse_args():
    hstr = "Model evaluation \n\n"\
           "Example 1: Evaluate the model with a single GPU: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/val.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \n\n"\
           "Example 2： Evaluate the model with multiple GPUs: \n"\
           "    export CUDA_VISIBLE_DEVICES=0,1 \n"\
           "    python -m paddle.distributed.launch tools/val.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        -o global.num_workers=2 \n\n"\
           "Example 3: Evaluate the model with test-time data augmentation: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/val.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        -o test.is_aug=True test.scales=0.75,1.0,1.25 test.flip_horizontal=True global.num_workers=2 \n\n"\
           "Example 4: Evaluate the model using sliding windows: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/val.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
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
        help="The path of trained weights to be loaded for evaluation.",
        type=str)
    parser.add_argument(
        '--opts',
        help="Specify key-value pairs to update configurations.",
        default=None,
        nargs='+')

    return parser.parse_args()


def main(args):
    assert args.config is not None, \
        "No configuration file has been specified. Please set `--config`."
    cfg = Config(args.config, opts=args.opts)
    global_cfg = cfg.global_cfg
    builder = SegBuilder(cfg)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_device(global_cfg['device'])

    # TODO refactor
    # Only support for the DeepLabv3+ model
    data_format = global_cfg['data_format']
    if data_format == 'NHWC':
        if cfg.dic['model']['type'] != 'DeepLabV3P':
            raise ValueError(
                "The 'NHWC' data format only support the DeepLabV3P model!")
        cfg.dic['model']['data_format'] = data_format
        cfg.dic['model']['backbone']['data_format'] = data_format
        loss_len = len(cfg.dic['loss']['types'])
        for i in range(loss_len):
            cfg.dic['loss']['types'][i]['data_format'] = data_format

    model = builder.model
    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info("Loaded trained weights successfully.")
    val_dataset = builder.val_dataset

    evaluate(
        model,
        val_dataset,
        num_workers=global_cfg['num_workers'],
        **cfg.test_cfg)


if __name__ == '__main__':
    args = parse_args()
    main(args)

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
from paddleseg.core import evaluate
from paddleseg.utils import get_sys_env, logger, utils


def parse_args():
    hstr = "Model Evaluation. \n\n"\
           "Example 1, evaluate model on single GPU: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/val.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \n\n"\
           "Example 2, evaluate model on multi GPUs: \n"\
           "    export CUDA_VISIBLE_DEVICES=0,1 \n"\
           "    python -m paddle.distributed.launch tools/val.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        -o global.num_workers=2 \n\n"\
           "Example 3, evaluate model with data augmentation: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/val.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        -o test.is_aug=True test.scales=0.75,1.0,1.25 test.flip_horizontal=True global.num_workers=2 \n\n"\
           "Example 4, evaluate model with slide windows: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/val.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        -o test.is_slide=True test.crop_size=256,256 test.stride=256,256 \n\n"\
           "Use `-o` or `--opts` to update key-value params in config file. Some common params are explained as follows:\n" \
           "    global.device       Set the running device. It should be cpu, gpu, xpu, npu or mlu.\n" \
           "    global.num_workers  Set the num workers to read and process images.\n" \
           "    test.is_aug         Whether to enable data augmentation. It should be True or False.\n" \
           "    test.scales         Set the resize scales in data augmentation. When test.is_aug=False, test.scales is invalid.\n" \
           "    test.flip_horizontal    Whether to flip horizontal in data augmentation. When test.is_aug=False, test.flip_horizontal is invalid.\n" \
           "    test.flip_vertical      Whether to flip vertical in data augmentation. When test.is_aug=False, test.flip_vertical is invalid.\n" \
           "    test.is_slide       Whether to test image with slide windows method. It should be True or False.\n" \
           "    test.crop_size      Set the crop size in data slide windows testing. When test.is_slide=False, test.crop_size is invalid.\n" \
           "    test.stride         Set the stride in slide windows testing. When test.is_slide=False, test.stride is invalid.\n"
    parser = argparse.ArgumentParser(
        description=hstr, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config', help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help='The path of model weight for evaluation',
        type=str)
    parser.add_argument(
        '-o',
        '--opts',
        help='Update the key-value pairs in config file.',
        nargs='+')
    return parser.parse_args()

    #TODO parse unknown params


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config.'
    cfg = Config(args.config, opts=args.opts)

    utils.show_env_info()
    utils.show_cfg_info(cfg)

    utils.set_seed(cfg.global_config('seed'))
    utils.set_device(cfg.global_config('device'))
    '''
    # Only support for the DeepLabv3+ model
    if args.data_format == 'NHWC':
        if cfg.dic['model']['type'] != 'DeepLabV3P':
            raise ValueError(
                'The "NHWC" data format only support the DeepLabV3P model!')
        cfg.dic['model']['data_format'] = args.data_format
        cfg.dic['model']['backbone']['data_format'] = args.data_format
        loss_len = len(cfg.dic['loss']['types'])
        for i in range(loss_len):
            cfg.dic['loss']['types'][i]['data_format'] = args.data_format
    '''

    model = cfg.model
    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    evaluate(
        model=model,
        eval_dataset=cfg.val_dataset,
        aug_eval=cfg.test_config('is_aug'),
        scales=cfg.test_config('scales'),
        flip_horizontal=cfg.test_config('flip_horizontal'),
        flip_vertical=cfg.test_config('flip_vertical'),
        is_slide=cfg.test_config('is_slide'),
        stride=cfg.test_config('stride'),
        crop_size=cfg.test_config('crop_size'),
        precision=cfg.global_config('precision'),
        amp_level=cfg.global_config('amp_level'),
        num_workers=cfg.global_config('num_workers'))


if __name__ == '__main__':
    args = parse_args()
    main(args)

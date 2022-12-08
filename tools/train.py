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
import random

import paddle
import numpy as np
import cv2

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import get_sys_env, logger, utils
from paddleseg.core import train


def parse_args():
    hstr = "Model Training. \n\n"\
           "Example 1, train model on single GPU: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/train.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        -o train.do_eval=True train.use_vdl=True train.save_interval=500 \\\n"\
           "            global.num_workers=2 global.save_dir=./output \n\n"\
           "Example 2, train model on multi GPUs: \n"\
           "    export CUDA_VISIBLE_DEVICES=0,1 \n"\
           "    python -m paddle.distributed.launch tools/train.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        -o train.do_eval=True train.use_vdl=True train.save_interval=500 \\\n"\
           "            global.num_workers=2 global.save_dir=./output \n\n" \
           "Example 3, resume training: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/train.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        -o train.do_eval=True train.use_vdl=True train.save_interval=500 train.resume_model=output/iter_500 \\\n"\
           "            global.num_workers=2 global.save_dir=./output/resume_training \n\n" \
           "Use `-o` or `--opts` to update key-value params in config file. Some common params are explained as follows:\n" \
           "    global.device       Set the running device. It should be cpu, gpu, xpu, npu or mlu.\n" \
           "    global.save_dir     Set the directory to save weights and log.\n" \
           "    global.num_workers  Set the num workers to read and process images.\n" \
           "    train.do_eval       Whether enable evaluation in training model. It should be True or False.\n" \
           "    train.use_vdl       Whether enable visualdl in training model. It should be True or False.\n" \
           "    train.resume_model  Resume training loads the files saved in this directory.\n"
    parser = argparse.ArgumentParser(
        description=hstr, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config', help="The path of config file.", type=str)
    parser.add_argument(
        '-o',
        '--opts',
        help='Update the key-value pairs in config file.',
        nargs='+')
    return parser.parse_args()

    #TODO parse unknown params for compatibility


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config.'
    cfg = Config(args.config, opts=args.opts)

    utils.show_env_info()
    utils.show_cfg_info(cfg)

    utils.set_seed(cfg.global_config('seed'))
    utils.set_device(cfg.global_config('device'))
    ''' TODO
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
    if cfg.global_config('device') == 'gpu' \
        and paddle.distributed.ParallelEnv().nranks > 1:
        model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    val_dataset = cfg.val_dataset if cfg.train_config("do_eval") else None
    test_config = {
        'aug_eval': cfg.test_config('is_aug'),
        'scales': cfg.test_config('scales'),
        'flip_horizontal': cfg.test_config('flip_horizontal'),
        'flip_vertical': cfg.test_config('flip_vertical'),
        'is_slide': cfg.test_config('is_slide'),
        'crop_size': cfg.test_config('crop_size'),
        'stride': cfg.test_config('stride'),
    }

    train(
        model=model,
        train_dataset=cfg.train_dataset,
        val_dataset=val_dataset,
        optimizer=cfg.optimizer,
        save_dir=cfg.global_config('save_dir'),
        iters=cfg.train_config('iters'),
        batch_size=cfg.train_config('batch_size'),
        resume_model=cfg.train_config('resume_model'),
        save_interval=cfg.train_config('save_interval'),
        log_iters=cfg.train_config('log_iters'),
        num_workers=cfg.global_config('num_workers'),
        use_vdl=cfg.train_config('use_vdl'),
        losses=cfg.loss,
        keep_checkpoint_max=cfg.train_config('keep_checkpoint_max'),
        test_config=test_config,
        precision=cfg.global_config('precision'),
        amp_level=cfg.global_config('amp_level'),
        profiler_options=cfg.train_config('profiler_options'),
        to_static_training=cfg.train_config("to_static_training"))


if __name__ == '__main__':
    args = parse_args()
    main(args)

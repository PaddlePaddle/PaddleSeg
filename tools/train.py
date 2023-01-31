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
from paddleseg.utils import utils
from paddleseg.core import train


def parse_args():
    hstr = "Model training \n\n"\
           "Example 1: Train a model with a single GPU: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/train.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        -o train.do_eval=True train.use_vdl=True train.save_interval=500 \\\n"\
           "            global.num_workers=2 global.save_dir=./output \n\n"\
           "Example 2: Train model with multiple GPUs: \n"\
           "    export CUDA_VISIBLE_DEVICES=0,1 \n"\
           "    python -m paddle.distributed.launch tools/train.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        -o train.do_eval=True train.use_vdl=True train.save_interval=500 \\\n"\
           "            global.num_workers=2 global.save_dir=./output \n\n" \
           "Example 3: Resume training from one checkpoint: \n"\
           "    export CUDA_VISIBLE_DEVICES=0 \n"\
           "    python tools/train.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        -o train.do_eval=True train.use_vdl=True train.save_interval=500 train.resume_model=output/iter_500 \\\n"\
           "            global.num_workers=2 global.save_dir=./output/resume_training \n\n" \
           "Use `-o` or `--opts` to overwrite key-value config items. Some common configurations are explained as follows:\n" \
           "    global.device       Set the running device. It should be 'cpu', 'gpu', 'xpu', 'npu', or 'mlu'.\n" \
           "    global.save_dir     Set the directory to save weights and logs.\n" \
           "    global.num_workers  Set the number of workers to read and process images.\n" \
           "    train.do_eval       Whether or not to enable periodical evaluation during training. It should be either True or False.\n" \
           "    train.resume_model  Set the path of the checkpoint to resume training from.\n"

    parser = argparse.ArgumentParser(
        description=hstr, formatter_class=argparse.RawTextHelpFormatter)

    # Common params
    parser.add_argument('--config', help="The path of config file.", type=str)
    parser.add_argument(
        '--use_vdl',
        help="Whether or not to enable VisualDL during training.",
        action='store_true')
    parser.add_argument(
        '--profiler_options',
        type=str,
        help="The options for the training profiler. If `profiler_options` is not None, the ' \
            'profiler will be enabled. Refer to `paddleseg/utils/train_profiler.py` for details."
    )
    parser.add_argument(
        '--opts',
        help="Specify key-value pairs to update configurations.",
        nargs='+')

    return parser.parse_args()


def main(args):
    assert args.config is not None, \
        "No configuration file has been specified. Please set `--config`."
    cfg = Config(args.config, opts=args.opts)
    runtime_cfg = cfg.runtime_cfg
    builder = SegBuilder(cfg)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_seed(runtime_cfg['seed'])
    utils.set_device(runtime_cfg['device'])
    utils.set_cv2_num_threads(runtime_cfg['num_workers'])

    # TODO refactor
    # Only support for the DeepLabv3+ model
    data_format = runtime_cfg['data_format']
    if data_format == 'NHWC':
        if cfg.dic['model']['type'] != 'DeepLabV3P':
            raise ValueError(
                'The "NHWC" data format only support the DeepLabV3P model!')
        cfg.dic['model']['data_format'] = data_format
        cfg.dic['model']['backbone']['data_format'] = data_format
        loss_len = len(cfg.dic['loss']['types'])
        for i in range(loss_len):
            cfg.dic['loss']['types'][i]['data_format'] = data_format

    model = utils.convert_sync_batchnorm(builder.model, runtime_cfg['device'])

    train_dataset = builder.train_dataset
    # TODO refactor
    if runtime_cfg['repeats'] > 1:
        train_dataset.file_list *= runtime_cfg['repeats']
    val_dataset = builder.val_dataset if runtime_cfg['do_eval'] else None
    optimizer = builder.optimizer
    loss = builder.loss

    train(
        model,
        train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        save_dir=runtime_cfg['save_dir'],
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        resume_model=runtime_cfg['resume_model'],
        save_interval=runtime_cfg['save_interval'],
        log_iters=runtime_cfg['log_iters'],
        num_workers=runtime_cfg['num_workers'],
        use_vdl=args.use_vdl,
        losses=loss,
        keep_checkpoint_max=runtime_cfg['keep_checkpoint_max'],
        test_config=cfg.test_cfg,
        precision=runtime_cfg['precision'],
        amp_level=runtime_cfg['amp_level'],
        profiler_options=args.profiler_options,
        to_static_training=cfg.to_static_training)


if __name__ == '__main__':
    args = parse_args()
    main(args)

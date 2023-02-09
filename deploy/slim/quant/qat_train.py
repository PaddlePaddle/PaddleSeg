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

import paddle
from paddleslim import QAT

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.core import train
from paddleseg.utils import logger, utils
from qat_config import quant_config
"""
Apply quantization to segmentation model.
NOTE: Only conv2d and linear in backbone are quantized.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Quantization aware training')

    parser.add_argument('--config', help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help="The path of pretrained model.",
        type=str,
        default=None)
    parser.add_argument(
        '--opts',
        help="Specify key-value pairs to update configurations.",
        nargs='+')

    return parser.parse_args()


def skip_quant(model):
    """
    If the model has backbone and head, we skip quantizing the conv2d and linear ops
    that belongs the head.
    """
    if not hasattr(model, 'backbone'):
        logger.info("Quantize all target ops")
        return

    logger.info("Quantize all target ops in backbone")
    for name, cur_layer in model.named_sublayers():
        if isinstance(cur_layer, (paddle.nn.Conv2D, paddle.nn.Linear)) \
            and 'backbone' not in name:
            cur_layer.skip_quant = True


def main(args):
    if not args.config:
        raise RuntimeError("No configuration file has been specified.")

    cfg = Config(args.config, opts=args.opts)
    runtime_cfg = cfg.runtime_cfg
    builder = SegBuilder(cfg)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_seed(runtime_cfg['seed'])
    utils.set_device(runtime_cfg['device'])
    utils.set_cv2_num_threads(runtime_cfg['num_workers'])

    model = builder.model
    train_dataset = builder.train_dataset
    val_dataset = builder.val_dataset if runtime_cfg['do_eval'] else None
    losses = builder.loss
    optimizer = builder.optimizer

    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info("Loaded trained weights successfully.")

    skip_quant(model)
    quantizer = QAT(config=quant_config)
    quantizer.quantize(model)
    logger.info("The model has been successfully quantized.")

    if runtime_cfg['resume_model'] is not None:
        logger.warning(
            "`resume_model` is not None, but it will not be used in QAT.")

    train(
        model,
        train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        save_dir=runtime_cfg['save_dir'],
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        resume_model=None,
        save_interval=runtime_cfg['save_interval'],
        log_iters=runtime_cfg['log_iters'],
        num_workers=runtime_cfg['num_workers'],
        use_vdl=runtime_cfg['use_vdl'],
        losses=losses,
        keep_checkpoint_max=runtime_cfg['keep_checkpoint_max'])


if __name__ == '__main__':
    args = parse_args()
    main(args)

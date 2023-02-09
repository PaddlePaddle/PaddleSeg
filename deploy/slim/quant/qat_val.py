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

from paddleslim import QAT

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.core import evaluate
from paddleseg.utils import logger, utils
from qat_config import quant_config
from qat_train import skip_quant


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of QAT models')

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
    if not args.config:
        raise RuntimeError("No configuration file has been specified.")

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
    val_dataset = builder.val_dataset

    skip_quant(model)
    quantizer = QAT(config=quant_config)
    quantizer.quantize(model)
    logger.info("The model has been successfully quantized.")

    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info("Loaded trained weights successfully.")

    evaluate(
        model,
        val_dataset,
        num_workers=global_cfg['num_workers'],
        **cfg.test_cfg)


if __name__ == '__main__':
    args = parse_args()
    main(args)

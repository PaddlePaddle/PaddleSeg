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
import os
import shutil

import yaml

import paddle
from paddleslim.dygraph.quant import QAT
from paddleseg.cvlibs.config import Config
from paddleseg.core.val import evaluate
from paddleseg.core.train import train
from paddleseg.utils import get_sys_env, logger


def parse_args():
    parser = argparse.ArgumentParser(description='Model pruning')
    # params of pruning
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=None)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=None)
    parser.add_argument(
        '--retraining_iters',
        dest='retraining_iters',
        help='Number of iterations of retraining.',
        type=int,
        default=None,
        required=True)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for evaluation',
        type=str,
        default=None)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)

    return parser.parse_args()


def get_quant_config():
    quant_config = {
        # weight preprocess type, default is None and no preprocessing is performed.
        'weight_preprocess_type': None,
        # activation preprocess type, default is None and no preprocessing is performed.
        'activation_preprocess_type': None,
        # weight quantize type, default is 'channel_wise_abs_max'
        'weight_quantize_type': 'channel_wise_abs_max',
        # activation quantize type, default is 'moving_average_abs_max'
        'activation_quantize_type': 'moving_average_abs_max',
        # weight quantize bit num, default is 8
        'weight_bits': 8,
        # activation quantize bit num, default is 8
        'activation_bits': 8,
        # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
        'dtype': 'int8',
        # window size for 'range_abs_max' quantization. default is 10000
        'window_size': 10000,
        # The decay coefficient of moving average, default is 0.9
        'moving_rate': 0.9,
        # for dygraph quantization, layers of type in quantizable_layer_type will be quantized
        'quantizable_layer_type': ['Conv2D', 'Linear'],
    }
    return quant_config


def main(args):
    env_info = get_sys_env()

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'
    paddle.set_device(place)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cfg = Config(
        args.cfg,
        iters=args.retraining_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate)

    train_dataset = cfg.train_dataset
    if not train_dataset:
        raise RuntimeError(
            'The training dataset is not specified in the configuration file.')

    val_dataset = cfg.val_dataset
    if not val_dataset:
        raise RuntimeError(
            'The validation dataset is not specified in the configuration file.'
        )
    os.environ['PADDLESEG_EXPORT_STAGE'] = 'True'
    net = cfg.model

    if args.model_path:
        para_state_dict = paddle.load(args.model_path)
        net.set_dict(para_state_dict)
        logger.info('Loaded trained params of model successfully')

    logger.info('Step 1/2: Start to quantify the model...')
    quantizer = QAT(config=get_quant_config())
    quantizer.quantize(net)
    logger.info('Model quantification completed.')

    logger.info('Step 2/2: Start retraining the quantized model.')
    train(
        net,
        train_dataset,
        optimizer=cfg.optimizer,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        losses=cfg.loss)

    evaluate(net, val_dataset)

    if paddle.distributed.get_rank() == 0:
        save_path = os.path.join(args.save_dir, 'model')
        input_var = paddle.ones([1] + list(val_dataset[0][0].shape))
        quantizer.save_quantized_model(net, save_path, input_spec=[input_var])

        yml_file = os.path.join(args.save_dir, 'deploy.yaml')
        with open(yml_file, 'w') as file:
            transforms = cfg.dic['val_dataset']['transforms']
            data = {
                'Deploy': {
                    'transforms': transforms,
                    'model': 'model.pdmodel',
                    'params': 'model.pdiparams'
                }
            }
            yaml.dump(data, file)

        ckpt = os.path.join(args.save_dir, f'iter_{args.retraining_iters}')
        if os.path.exists(ckpt):
            shutil.rmtree(ckpt)

    logger.info(
        f'Model retraining complete. The quantized model is saved in {args.save_dir}.'
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)

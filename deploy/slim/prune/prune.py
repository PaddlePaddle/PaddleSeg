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
from functools import partial

import yaml

import paddle
from paddleslim.dygraph import L1NormFilterPruner
from paddleslim.analysis import dygraph_flops
from paddleseg.cvlibs.config import Config
from paddleseg.core.val import evaluate
from paddleseg.core.train import train
from paddleseg.utils import get_sys_env, logger


def parse_args():
    parser = argparse.ArgumentParser(description='Model pruning')
    # params of pruning
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", type=str, default=None)
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
        "--pruning_ratio",
        dest="pruning_ratio",
        help="The ratio of model pruning.",
        type=float,
        default=None,
        required=True)
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


def eval_fn(net, eval_dataset, num_workers):
    miou, _, _, _, _ = evaluate(
        net, eval_dataset, num_workers=num_workers, print_detail=False)
    return miou


def export_model(net, cfg, save_dir):
    net.forward = paddle.jit.to_static(net.forward)
    input_shape = [1] + list(cfg.val_dataset[0][0].shape)
    input_var = paddle.ones(input_shape)
    out = net(input_var)

    save_path = os.path.join(save_dir, 'model')
    paddle.jit.save(net, save_path, input_spec=[input_var])

    yml_file = os.path.join(save_dir, 'deploy.yaml')
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


def main(args):
    env_info = get_sys_env()

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'
    paddle.set_device(place)

    if not (0.0 < args.pruning_ratio < 1.0):
        raise RuntimeError(
            'The model pruning rate must be in the range of (0, 1).')

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
            'The validation dataset is not specified in the c;onfiguration file.'
        )
    os.environ['PADDLESEG_EXPORT_STAGE'] = 'True'
    net = cfg.model

    if args.model_path:
        para_state_dict = paddle.load(args.model_path)
        net.set_dict(para_state_dict)
        logger.info('Loaded trained params of model successfully')

    logger.info(
        'Step 1/3: Start calculating the sensitivity of model parameters...')
    sample_shape = [1] + list(train_dataset[0][0].shape)
    sen_file = os.path.join(args.save_dir, 'sen.pickle')
    pruner = L1NormFilterPruner(net, sample_shape)
    pruner.sensitive(
        eval_func=partial(eval_fn, net, val_dataset, args.num_workers),
        sen_file=sen_file)
    logger.info(
        f'The sensitivity calculation of model parameters is complete. The result is saved in {sen_file}.'
    )

    flops = dygraph_flops(net, sample_shape)
    logger.info(
        f'Step 2/3: Start to prune the model, the ratio of pruning is {args.pruning_ratio}. FLOPs before pruning: {flops}.'
    )

    # Avoid the bug when pruning conv2d with small channel number.
    # Remove this code after PaddleSlim 2.1 is available.
    # Related issue: https://github.com/PaddlePaddle/PaddleSlim/issues/674.
    skips = []
    for param in net.parameters():
        if param.shape[0] <= 8:
            skips.append(param.name)

    pruner.sensitive_prune(args.pruning_ratio, skip_vars=skips)
    flops = dygraph_flops(net, sample_shape)
    logger.info(f'Model pruning completed. FLOPs after pruning: {flops}.')

    logger.info(f'Step 3/3: Start retraining the model.')
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
        export_model(net, cfg, args.save_dir)

        ckpt = os.path.join(args.save_dir, f'iter_{args.retraining_iters}')
        if os.path.exists(ckpt):
            shutil.rmtree(ckpt)

    logger.info(f'Model retraining finish. Model is saved in {args.save_dir}')


if __name__ == '__main__':
    args = parse_args()
    main(args)

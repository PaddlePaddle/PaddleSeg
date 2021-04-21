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

import paddle
import yaml

from paddleseg.cvlibs import Config
from paddleseg.utils import logger

from datasets.humanseg import HumanSeg
from models import *


def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    # params of training
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default=None,
        type=str,
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
        '--add_argmax',
        dest='add_argmax',
        help='The path of model for evaluation',
        type=bool,
        default=False)
    parser.add_argument(
        '--add_softmax',
        dest='add_softmax',
        help='The path of model for evaluation',
        type=bool,
        default=False)

    return parser.parse_args()


class NewNet(paddle.nn.Layer):
    def __init__(self, net, add_argmax=False, add_softmax=False):
        super().__init__()
        self.net = net
        self.add_argmax = add_argmax
        self.add_softmax = add_softmax

    def forward(self, x):
        outs = self.net(x)
        new_outs = []
        for out in outs:
            if self.add_softmax:
                out = paddle.nn.functional.softmax(out, axis=1)
            if self.add_argmax:
                out = paddle.argmax(out, axis=1)
            new_outs.append(out)
        return new_outs


def main(args):
    os.environ['PADDLESEG_EXPORT_STAGE'] = 'True'
    cfg = Config(args.cfg)
    net = cfg.model

    if args.model_path:
        para_state_dict = paddle.load(args.model_path)
        net.set_dict(para_state_dict)
        logger.info('Loaded trained params of model successfully.')

    if args.add_argmax or args.add_softmax:
        new_net = NewNet(net, args.add_argmax, args.add_softmax)
    else:
        new_net = net

    new_net.eval()
    new_net = paddle.jit.to_static(
        new_net,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, 3, None, None], dtype='float32')
        ])
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(new_net, save_path)

    yml_file = os.path.join(args.save_dir, 'deploy.yaml')
    with open(yml_file, 'w') as file:
        transforms = cfg.export_config.get('transforms', [{
            'type': 'Normalize'
        }])
        data = {
            'Deploy': {
                'transforms': transforms,
                'model': 'model.pdmodel',
                'params': 'model.pdiparams'
            }
        }
        yaml.dump(data, file)

    logger.info(f'Model is saved in {args.save_dir}.')


if __name__ == '__main__':
    args = parse_args()
    main(args)

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


def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument(
        "--config", help="The config file.", type=str, required=True)
    parser.add_argument(
        '--model_path', help='The path of model for export', type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the exported model',
        type=str,
        default='./output/inference_model')
    parser.add_argument(
        '--output_op',
        choices=['argmax', 'softmax', 'none'],
        default="argmax",
        help="Select which op to be appended to output result, default: argmax")
    parser.add_argument(
        '--without_argmax',
        help='Do not add the argmax operation at the end of the network. [Deprecated]',
        action='store_true')
    parser.add_argument(
        '--with_softmax',
        help='Add the softmax operation at the end of the network. [Deprecated]',
        action='store_true')
    parser.add_argument(
        "--input_shape",
        nargs='+',
        help="Export the model with fixed input shape, such as 1 3 1024 1024.",
        type=int,
        default=None)

    return parser.parse_args()


class SavedSegmentationNet(paddle.nn.Layer):
    def __init__(self, net, output_op):
        super().__init__()
        self.net = net
        self.output_op = output_op
        assert output_op in ['argmax', 'softmax'], \
            "output_op should in ['argmax', 'softmax']"

    def forward(self, x):
        outs = self.net(x)

        new_outs = []
        for out in outs:
            if self.output_op == 'argmax':
                out = paddle.argmax(out, axis=1, dtype='int32')
            elif self.output_op == 'softmax':
                out = paddle.nn.functional.softmax(out, axis=1)
            new_outs.append(out)
        return new_outs


def main(args):
    os.environ['PADDLESEG_EXPORT_STAGE'] = 'True'
    cfg = Config(args.config)
    cfg.check_sync_info()
    net = cfg.model

    if args.model_path is not None:
        para_state_dict = paddle.load(args.model_path)
        net.set_dict(para_state_dict)
        logger.info('Loaded trained params of model successfully.')

    if args.input_shape is None:
        shape = [None, 3, None, None]
    else:
        shape = args.input_shape

    output_op = args.output_op
    if args.without_argmax:
        logger.warning(
            '--without_argmax will be deprecated, please use --output_op')
        output_op = 'none'
    if args.with_softmax:
        logger.warning(
            '--with_softmax will be deprecated, please use --output_op')
        output_op = 'softmax'

    new_net = net if output_op == 'none' else SavedSegmentationNet(net,
                                                                   output_op)
    new_net.eval()
    new_net = paddle.jit.to_static(
        new_net,
        input_spec=[paddle.static.InputSpec(
            shape=shape, dtype='float32')])

    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(new_net, save_path)

    yml_file = os.path.join(args.save_dir, 'deploy.yaml')
    with open(yml_file, 'w') as file:
        transforms = cfg.export_config.get('transforms', [{
            'type': 'Normalize'
        }])
        output_dtype = 'int32' if output_op == 'argmax' else 'float32'
        data = {
            'Deploy': {
                'model': 'model.pdmodel',
                'params': 'model.pdiparams',
                'transforms': transforms,
                'input_shape': shape,
                'output_op': output_op,
                'output_dtype': output_dtype
            }
        }
        yaml.dump(data, file)

    logger.info(f'The inference model is saved in {args.save_dir}')

    logger.warning("This `export.py` will be removed in version 2.8, "
                   "please use `tools/export.py`.")


if __name__ == '__main__':
    args = parse_args()
    main(args)

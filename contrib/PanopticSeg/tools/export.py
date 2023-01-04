# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddleseg.utils import logger

from paddlepanseg.cvlibs import Config


class PanSegInferNetWrapper(paddle.nn.Layer):
    def __init__(self, net, postprocessor):
        super().__init__()
        self.net = net
        self.postprocessor = postprocessor

    def forward(self, x):
        pred = self.net(x)
        # XXX: Currently, pass an empty sample_dict to postprocessor
        out = self.postprocessor({}, pred)
        return [out['pan_pred']]


def parse_export_args(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Model export")
    parser.add_argument(
        '--config', help="Config file.", type=str, required=True)
    parser.add_argument(
        '--model_path', help="Path of the model for export.", type=str)
    parser.add_argument(
        '--save_dir',
        help="Directory to save the exported model.",
        type=str,
        default="./output/inference_model")
    parser.add_argument(
        '--input_shape',
        nargs='+',
        help="Export the model with fixed input shape, such as 1 3 1024 1024.",
        type=int,
        default=None)

    return parser.parse_args(*args, **kwargs)


def export_with_args(args):
    os.environ['PADDLESEG_EXPORT_STAGE'] = 'True'
    cfg = Config(args.config)
    net = cfg.model
    postprocessor = cfg.postprocessor

    if args.model_path is not None:
        para_state_dict = paddle.load(args.model_path)
        net.set_dict(para_state_dict)
        logger.info("Params are successfully loaded.")

    if args.input_shape is None:
        shape = [None, 3, None, None]
    else:
        shape = args.input_shape

    net = PanSegInferNetWrapper(net, postprocessor)
    net.eval()
    net = paddle.jit.to_static(
        net,
        input_spec=[paddle.static.InputSpec(
            shape=shape, dtype='float32')])

    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(net, save_path)

    yml_file = os.path.join(args.save_dir, 'deploy.yaml')
    with open(yml_file, 'w') as file:
        transforms = cfg.export_config.get('transforms', [{
            'type': 'Normalize'
        }])
        data = {
            'Deploy': {
                'model': 'model.pdmodel',
                'params': 'model.pdiparams',
                'transforms': transforms,
                'input_shape': shape
            }
        }
        yaml.dump(data, file)

    logger.info(f"The inference model is exported to {args.save_dir}.")


if __name__ == '__main__':
    args = parse_export_args()
    export_with_args(args)

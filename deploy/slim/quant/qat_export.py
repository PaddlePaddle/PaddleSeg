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

import yaml
import paddle
from paddleslim import QAT

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import logger, utils
from paddleseg.deploy.export import WrappedModel
from qat_config import quant_config
from qat_train import skip_quant


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export quantized model for inference')
    parser.add_argument('--config', help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help="The path of trained weights for exporting inference model.",
        type=str)
    parser.add_argument(
        '--save_dir',
        help="The directory for saving the exported inference model.",
        type=str,
        default='./output/inference_model')
    parser.add_argument(
        "--input_shape",
        nargs='+',
        help="Export the model with a fixed input shape, e.g., `--input_shape 1 3 1024 1024`.",
        type=int,
        default=None)
    parser.add_argument(
        '--output_op',
        choices=['argmax', 'softmax', 'none'],
        default="argmax",
        help="Select the operator to be appended to the inference model. Default: argmax. "
        "In PaddleSeg, the output of a trained model is logits (H*C*H*W). We can apply argmax or"
        "softmax to the logits in different practice.")

    return parser.parse_args()


def main(args):
    assert args.config is not None, \
        "No configuration file has been specified. Please set `--config`."

    cfg = Config(args.config)
    builder = SegBuilder(cfg)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    os.environ['PADDLESEG_EXPORT_STAGE'] = 'True'

    # Quantize model
    net = builder.model
    skip_quant(net)
    quantizer = QAT(config=quant_config)
    quantizer.quantize(net)
    logger.info("The model has been successfully quantized.")

    if args.model_path is not None:
        utils.load_entire_model(net, args.model_path)
        logger.info("Loaded trained weights successfully.")

    output_op = args.output_op

    new_net = net if output_op == 'none' else WrappedModel(net, output_op)

    new_net.eval()
    save_path = os.path.join(args.save_dir, 'model')
    input_spec = [
        paddle.static.InputSpec(
            shape=[None, 3, None, None], dtype='float32')
    ]
    quantizer.save_quantized_model(new_net, save_path, input_spec=input_spec)

    yml_file = os.path.join(args.save_dir, 'deploy.yaml')
    with open(yml_file, 'w') as file:
        transforms = cfg.val_dataset_cfg.get('transforms', [{
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

    logger.info(f"The quantized inference model is saved in {args.save_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)

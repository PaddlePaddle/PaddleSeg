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

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import logger, utils
from paddleseg.deploy.export import WrappedModel


def parse_args():
    hstr = "Export model for inference \n\n"\
           "Example 1: Export inference model with dynamic shape: \n"\
           "    python tools/export.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        --save_dir output/inference_model \n\n"\
           "Example 2, export inference model with a fix shape: \n"\
           "    python tools/export.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        --save_dir output/inference_model \\\n"\
           "        --input_shape 1 3 512 512 \n\n"

    parser = argparse.ArgumentParser(
        description=hstr, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config', help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help="The path of trained weights for exporting inference model.",
        type=str)
    parser.add_argument(
        '--save_dir',
        help="The directory for saving the exported inference model.",
        type=str,
        default="./output/inference_model")
    parser.add_argument(
        '--input_shape',
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

    # Save model
    model = builder.model
    if args.model_path is not None:
        state_dict = paddle.load(args.model_path)
        model.set_dict(state_dict)
        logger.info("Loaded trained weights successfully.")
    if args.output_op != 'none':
        model = WrappedModel(model, args.output_op)

    shape = [None, 3, None, None] if args.input_shape is None \
        else args.input_shape
    input_spec = [paddle.static.InputSpec(shape=shape, dtype='float32')]
    model.eval()
    model = paddle.jit.to_static(model, input_spec=input_spec)
    paddle.jit.save(model, os.path.join(args.save_dir, 'model'))

    # Save deploy.yaml
    val_dataset_cfg = cfg.val_dataset_cfg
    assert val_dataset_cfg != {}, "`val_dataset` is not specified in the configuration file."
    transforms = val_dataset_cfg.get('transforms', None)
    output_dtype = 'int32' if args.output_op == 'argmax' else 'float32'

    # TODO add test config
    deploy_info = {
        'Deploy': {
            'model': 'model.pdmodel',
            'params': 'model.pdiparams',
            'transforms': transforms,
            'input_shape': shape,
            'output_op': args.output_op,
            'output_dtype': output_dtype
        }
    }
    msg = "\n---------------Deploy Information---------------\n"
    msg += str(yaml.dump(deploy_info))
    logger.info(msg)

    yml_file = os.path.join(args.save_dir, 'deploy.yaml')
    with open(yml_file, 'w') as file:
        yaml.dump(deploy_info, file)

    logger.info(f"The inference model is saved in {args.save_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)

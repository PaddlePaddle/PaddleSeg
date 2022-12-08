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
    hstr = "Model Export. \n\n"\
           "Example 1, export inference model with dynamic shape: \n"\
           "    python tools/export.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        --save_dir output/inference_model \n\n"\
           "Example 2, export inference model with fix shape: \n"\
           "    python tools/export.py \\\n"\
           "        --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \\\n"\
           "        --model_path output/best_model/model.pdparams \\\n"\
           "        --save_dir output/inference_model \\\n"\
           "        --input_shape 1 3 512 512 \n\n"
    parser = argparse.ArgumentParser(
        description=hstr, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config", help="The path of config file.", type=str, required=True)
    parser.add_argument(
        '--model_path',
        help='The path of model weights, e.g., `--model_path output/best_model/model.pdparams`',
        type=str)
    parser.add_argument(
        '--save_dir',
        help='Set the directory to save inference model',
        type=str,
        default='./output/inference_model')
    parser.add_argument(
        '--output_op',
        choices=['argmax', 'softmax', 'none'],
        default="argmax",
        help="Select the op to be appended to the last of inference model, default: argmax."
        "In PaddleSeg, the output of trained model is logit (H*C*H*W). We can apply argmax and"
        "softmax op to the logit according the actual situation.")
    parser.add_argument(
        "--input_shape",
        nargs='+',
        help="Export the model with fixed input shape, e.g., `--input_shape 1 3 1024 1024`.",
        type=int)

    return parser.parse_args()


class WrappedModel(paddle.nn.Layer):
    def __init__(self, model, output_op):
        super().__init__()
        self.model = model
        self.output_op = output_op
        assert output_op in ['argmax', 'softmax'], \
            "output_op should in ['argmax', 'softmax']"

    def forward(self, x):
        outs = self.model(x)

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
    model = cfg.model

    if args.model_path is not None:
        para_state_dict = paddle.load(args.model_path)
        model.set_dict(para_state_dict)
        logger.info('Loaded trained params of model successfully.')

    if args.output_op != 'none':
        model = WrappedModel(model, args.output_op)

    if args.input_shape is None:
        shape = [None, 3, None, None]
    else:
        shape = args.input_shape
    input_spec = [paddle.static.InputSpec(shape=shape, dtype='float32')]

    model.eval()
    model = paddle.jit.to_static(model, input_spec=input_spec)
    paddle.jit.save(model, os.path.join(args.save_dir, 'model'))

    val_dataset_config = cfg.val_dataset_config
    assert val_dataset_config != {}, 'No val_dataset specified in the configuration file.'
    transforms = val_dataset_config.get('transforms', None)
    assert transforms is not None, 'No transforms specified in val_dataset.'
    output_dtype = 'int32' if args.output_op == 'argmax' else 'float32'
    test = {
        'is_aug': cfg.test_config('is_aug'),
        'scales': cfg.test_config('scales'),
        'flip_horizontal': cfg.test_config('flip_horizontal'),
        'flip_vertical': cfg.test_config('flip_vertical'),
        'is_slide': cfg.test_config('is_slide'),
        'crop_size': cfg.test_config('crop_size'),
        'stride': cfg.test_config('stride'),
    }

    deploy_info = {
        'Deploy': {
            'model': 'model.pdmodel',
            'params': 'model.pdiparams',
            'transforms': transforms,
            'test': test,
            'input_shape': shape,
            'output_op': args.output_op,
            'output_dtype': output_dtype
        }
    }
    msg = '\n---------------Deploy Information---------------\n'
    msg += str(yaml.dump(deploy_info))
    logger.info(msg)

    yml_file = os.path.join(args.save_dir, 'deploy.yaml')
    with open(yml_file, 'w') as file:
        yaml.dump(deploy_info, file)

    logger.info(f'The inference model is saved in {args.save_dir}')


if __name__ == '__main__':
    args = parse_args()
    main(args)

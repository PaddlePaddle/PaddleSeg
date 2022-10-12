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
"""
Export the Paddle model to ONNX.

Usage:
    python tools/export_onnx.py \
        --config configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml

Note:
* Some models are not supported exporting to ONNX.
* Some ONNX models are not supportd deploying by TRT.
"""

import argparse
import codecs
import os
import sys
import time

import numpy as np
from tqdm import tqdm

import paddle
import onnx
import onnxruntime

from paddleseg.cvlibs import Config
from paddleseg.utils import logger, utils


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--config", help="The config file.", type=str)
    parser.add_argument(
        "--model_path", help="The pretrained weights file.", type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the predict result.',
        type=str,
        default='./output/tmp')
    parser.add_argument('--width', help='width', type=int, default=512)
    parser.add_argument('--height', help='height', type=int, default=512)
    parser.add_argument(
        '--print_model', action='store_true', help='print model to log')
    return parser.parse_args()


def run_paddle(paddle_model, input_data):
    paddle_model.eval()
    paddle_outs = paddle_model(paddle.to_tensor(input_data))
    out = paddle_outs[0].numpy()
    if out.ndim == 3:
        out = out[np.newaxis, :]
    return out


def check_and_run_onnx(onnx_model_path, input_data):
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print('The onnx model has been checked.')

    ort_sess = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {ort_sess.get_inputs()[0].name: input_data}
    ort_outs = ort_sess.run(None, ort_inputs)
    print("The onnx model has been predicted by ONNXRuntime.")

    return ort_outs[0]


def export_onnx(args):
    cfg = Config(args.config)
    cfg.check_sync_info()
    model = cfg.model
    if args.model_path is not None:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')

    model.eval()
    if args.print_model:
        print(model)

    input_shape = [1, 3, args.height, args.width]
    print("input shape:", input_shape)
    input_data = np.random.random(input_shape).astype('float32')
    model_name = os.path.basename(args.config).split(".")[0]

    paddle_out = run_paddle(model, input_data)
    print("out shape:", paddle_out.shape)
    print("The paddle model has been predicted by PaddlePaddle.\n")

    input_spec = paddle.static.InputSpec(input_shape, 'float32', 'x')
    onnx_model_path = os.path.join(args.save_dir, model_name + "_model")
    paddle.onnx.export(
        model, onnx_model_path, input_spec=[input_spec], opset_version=11)
    print("Completed export onnx model.\n")

    onnx_model_path = onnx_model_path + ".onnx"
    onnx_out = check_and_run_onnx(onnx_model_path, input_data)
    assert onnx_out.shape == paddle_out.shape
    np.testing.assert_allclose(onnx_out, paddle_out, rtol=0, atol=1e-03)
    print("The paddle and onnx models have the same outputs.\n")


if __name__ == '__main__':
    args = parse_args()
    export_onnx(args)

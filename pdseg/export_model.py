# coding: utf8
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import pprint
import cv2
import argparse
import numpy as np
import paddle.fluid as fluid

from utils.config import cfg
from models.model_builder import build_model
from models.model_builder import ModelPhase


def parse_args():
    parser = argparse.ArgumentParser(
        description='PaddleSeg Inference Model Exporter')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str)
    parser.add_argument(
        'opts',
        help='See utils/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def export_inference_model(args):
    """
    Export PaddlePaddle inference model for prediction depolyment and serving.
    """
    print("Exporting inference model...")
    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    image, logit_out = build_model(
        infer_prog, startup_prog, phase=ModelPhase.PREDICT)

    # Use CPU for exporting inference model instead of GPU
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    infer_prog = infer_prog.clone(for_test=True)

    if os.path.exists(cfg.TEST.TEST_MODEL):
        fluid.io.load_params(exe, cfg.TEST.TEST_MODEL, main_program=infer_prog)
    else:
        print("TEST.TEST_MODEL diretory is empty!")
        exit(-1)

    fluid.io.save_inference_model(
        cfg.FREEZE.SAVE_DIR,
        feeded_var_names=[image.name],
        target_vars=[logit_out],
        executor=exe,
        main_program=infer_prog,
        model_filename=cfg.FREEZE.MODEL_FILENAME,
        params_filename=cfg.FREEZE.PARAMS_FILENAME)
    print("Inference model exported!")


def main():
    args = parse_args()
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.update_from_list(args.opts)
    cfg.check_and_infer()
    print(pprint.pformat(cfg))
    export_inference_model(args)


if __name__ == '__main__':
    main()

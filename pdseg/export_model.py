# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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


def export_inference_config():
    deploy_cfg = '''DEPLOY:
        USE_GPU : 1
        USE_PR : 0
        MODEL_PATH : "%s"
        MODEL_FILENAME : "%s"
        PARAMS_FILENAME : "%s"
        EVAL_CROP_SIZE : %s
        MEAN : %s
        STD : %s
        IMAGE_TYPE : "%s"
        NUM_CLASSES : %d
        CHANNELS : %d
        PRE_PROCESSOR : "SegPreProcessor"
        PREDICTOR_MODE : "ANALYSIS"
        BATCH_SIZE : 1
    ''' % (cfg.FREEZE.SAVE_DIR, cfg.FREEZE.MODEL_FILENAME,
           cfg.FREEZE.PARAMS_FILENAME, cfg.EVAL_CROP_SIZE, cfg.MEAN, cfg.STD,
           cfg.DATASET.IMAGE_TYPE, cfg.DATASET.NUM_CLASSES, len(cfg.STD))
    if not os.path.exists(cfg.FREEZE.SAVE_DIR):
        os.mkdir(cfg.FREEZE.SAVE_DIR)
    yaml_path = os.path.join(cfg.FREEZE.SAVE_DIR, 'deploy.yaml')
    with open(yaml_path, "w") as fp:
        fp.write(deploy_cfg)
    return yaml_path


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
        print('load test model:', cfg.TEST.TEST_MODEL)
        try:
            fluid.load(infer_prog, os.path.join(cfg.TEST.TEST_MODEL, 'model'),
                       exe)
        except:
            fluid.io.load_params(
                exe, cfg.TEST.TEST_MODEL, main_program=infer_prog)
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
    print("Exporting inference model config...")
    deploy_cfg_path = export_inference_config()
    print("Inference model saved : [%s]" % (deploy_cfg_path))


def main():
    args = parse_args()
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)
    cfg.check_and_infer()
    print(pprint.pformat(cfg))
    export_inference_model(args)


if __name__ == '__main__':
    main()

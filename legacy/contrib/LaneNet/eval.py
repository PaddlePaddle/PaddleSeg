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
# GPU memory garbage collection optimization flags
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"

import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(cur_path)[0])[0]
LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
SEG_PATH = os.path.join(LOCAL_PATH, "../../../")
sys.path.append(SEG_PATH)
sys.path.append(root_path)

import time
import argparse
import functools
import pprint
import cv2
import numpy as np
import paddle
import paddle.fluid as fluid

from utils.config import cfg
from pdseg.utils.timer import Timer, calculate_eta
from models.model_builder import build_model
from models.model_builder import ModelPhase
from reader import LaneNetDataset


def parse_args():
    parser = argparse.ArgumentParser(description='PaddleSeg model evalution')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str)
    parser.add_argument(
        '--use_gpu',
        dest='use_gpu',
        help='Use gpu or cpu',
        action='store_true',
        default=False)
    parser.add_argument(
        '--use_mpio',
        dest='use_mpio',
        help='Use multiprocess IO or not',
        action='store_true',
        default=False)
    parser.add_argument(
        'opts',
        help='See utils/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def evaluate(cfg, ckpt_dir=None, use_gpu=False, use_mpio=False, **kwargs):
    np.set_printoptions(precision=5, suppress=True)

    startup_prog = fluid.Program()
    test_prog = fluid.Program()

    dataset = LaneNetDataset(
        file_list=cfg.DATASET.VAL_FILE_LIST,
        mode=ModelPhase.TRAIN,
        shuffle=True,
        data_dir=cfg.DATASET.DATA_DIR)

    def data_generator():
        #TODO: check is batch reader compatitable with Windows
        if use_mpio:
            data_gen = dataset.multiprocess_generator(
                num_processes=cfg.DATALOADER.NUM_WORKERS,
                max_queue_size=cfg.DATALOADER.BUF_SIZE)
        else:
            data_gen = dataset.generator()

        for b in data_gen:
            yield b

    data_loader, pred, grts, masks, accuracy, fp, fn = build_model(
        test_prog, startup_prog, phase=ModelPhase.EVAL)

    data_loader.set_sample_generator(
        data_generator, drop_last=False, batch_size=cfg.BATCH_SIZE)

    # Get device environment
    places = fluid.cuda_places() if use_gpu else fluid.cpu_places()
    place = places[0]
    dev_count = len(places)
    print("#Device count: {}".format(dev_count))

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    test_prog = test_prog.clone(for_test=True)

    ckpt_dir = cfg.TEST.TEST_MODEL if not ckpt_dir else ckpt_dir

    if ckpt_dir is not None:
        print('load test model:', ckpt_dir)
        try:
            fluid.load(test_prog, os.path.join(ckpt_dir, 'model'), exe)
        except:
            fluid.io.load_params(exe, ckpt_dir, main_program=test_prog)

    # Use streaming confusion matrix to calculate mean_iou
    np.set_printoptions(
        precision=4, suppress=True, linewidth=160, floatmode="fixed")
    fetch_list = [
        pred.name, grts.name, masks.name, accuracy.name, fp.name, fn.name
    ]
    num_images = 0
    step = 0
    avg_acc = 0.0
    avg_fp = 0.0
    avg_fn = 0.0
    # cur_images = 0
    all_step = cfg.DATASET.TEST_TOTAL_IMAGES // cfg.BATCH_SIZE + 1
    timer = Timer()
    timer.start()
    data_loader.start()
    while True:
        try:
            step += 1
            pred, grts, masks, out_acc, out_fp, out_fn = exe.run(
                test_prog, fetch_list=fetch_list, return_numpy=True)

            avg_acc += np.mean(out_acc) * pred.shape[0]
            avg_fp += np.mean(out_fp) * pred.shape[0]
            avg_fn += np.mean(out_fn) * pred.shape[0]
            num_images += pred.shape[0]

            speed = 1.0 / timer.elapsed_time()

            print(
                "[EVAL]step={} accuracy={:.4f} fp={:.4f} fn={:.4f} step/sec={:.2f} | ETA {}"
                .format(step, avg_acc / num_images, avg_fp / num_images,
                        avg_fn / num_images, speed,
                        calculate_eta(all_step - step, speed)))

            timer.restart()
            sys.stdout.flush()
        except fluid.core.EOFException:
            break

    print("[EVAL]#image={} accuracy={:.4f} fp={:.4f} fn={:.4f}".format(
        num_images, avg_acc / num_images, avg_fp / num_images,
        avg_fn / num_images))

    return avg_acc / num_images, avg_fp / num_images, avg_fn / num_images


def main():
    args = parse_args()
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)
    cfg.check_and_infer()
    print(pprint.pformat(cfg))
    evaluate(cfg, **args.__dict__)


if __name__ == '__main__':
    main()

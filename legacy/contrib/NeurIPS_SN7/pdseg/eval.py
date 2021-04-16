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
import argparse
import pprint
import numpy as np
import paddle.fluid as fluid

from utils.config import cfg
from utils.timer import Timer, calculate_eta
from models.model_builder import build_model
from models.model_builder import ModelPhase
from reader import SegDataset
from metrics import ConfusionMatrix


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
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument(
        '--vis_dir',
        dest='vis_dir',
        help='visual save dir',
        type=str,
        default='vis_out/test_public')
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


def evaluate(cfg,
             ckpt_dir=None,
             use_gpu=False,
             vis=False,
             vis_dir='vis_out/test_public',
             use_mpio=False,
             **kwargs):
    np.set_printoptions(precision=5, suppress=True)

    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    dataset = SegDataset(
        file_list=cfg.DATASET.VAL_FILE_LIST,
        mode=ModelPhase.EVAL,
        data_dir=cfg.DATASET.DATA_DIR)

    fls = []
    with open(cfg.DATASET.VAL_FILE_LIST) as fr:
        for line in fr.readlines():
            fls.append(line.strip().split(' ')[0])
    if vis:
        assert cfg.VIS.VISINEVAL is True
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def data_generator():
        #TODO: check is batch reader compatitable with Windows
        if use_mpio:
            data_gen = dataset.multiprocess_generator(
                num_processes=cfg.DATALOADER.NUM_WORKERS,
                max_queue_size=cfg.DATALOADER.BUF_SIZE)
        else:
            data_gen = dataset.generator()

        for b in data_gen:
            if cfg.DATASET.INPUT_IMAGE_NUM == 1:
                yield b[0], b[1], b[2]
            else:
                yield b[0], b[1], b[2], b[3]

    data_loader, avg_loss, pred, grts, masks = build_model(
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

    if not os.path.exists(ckpt_dir):
        raise ValueError('The TEST.TEST_MODEL {} is not found'.format(ckpt_dir))

    if ckpt_dir is not None:
        print('load test model:', ckpt_dir)
        try:
            fluid.load(test_prog, os.path.join(ckpt_dir, 'model'), exe)
        except:
            fluid.io.load_params(exe, ckpt_dir, main_program=test_prog)

    # Use streaming confusion matrix to calculate mean_iou
    np.set_printoptions(
        precision=4, suppress=True, linewidth=160, floatmode="fixed")
    class_num = cfg.DATASET.NUM_CLASSES
    conf_mat = ConfusionMatrix(class_num, streaming=True)
    fetch_list = [avg_loss.name, pred.name, grts.name, masks.name]
    num_images = 0
    step = 0
    all_step = cfg.DATASET.TEST_TOTAL_IMAGES // cfg.BATCH_SIZE + 1
    timer = Timer()
    timer.start()
    data_loader.start()
    cnt = 0
    while True:
        try:
            step += 1
            loss, pred, grts, masks = exe.run(
                test_prog, fetch_list=fetch_list, return_numpy=True)
            if vis:
                preds = np.array(pred, dtype=np.float32)
                for j in range(preds.shape[0]):
                    if cnt > len(fls): continue
                    name = fls[cnt].split('/')[-1].split('.')[0]
                    p = np.squeeze(preds[j])
                    np.save(os.path.join(vis_dir, name + '.npy'), p)
                    cnt += 1
                print('vis %d npy... (%d tif sample)' % (cnt, cnt // 36))
                continue

            loss = np.mean(np.array(loss))

            num_images += pred.shape[0]
            conf_mat.calculate(pred, grts, masks)
            _, iou = conf_mat.mean_iou()
            _, acc = conf_mat.accuracy()
            fwiou = conf_mat.frequency_weighted_iou()

            speed = 1.0 / timer.elapsed_time()

            print(
                "[EVAL]step={} loss={:.5f} acc={:.4f} IoU={:.4f} FWIoU={:.4f} step/sec={:.2f} | ETA {}"
                .format(step, loss, acc, iou, fwiou, speed,
                        calculate_eta(all_step - step, speed)))
            timer.restart()
            sys.stdout.flush()
        except fluid.core.EOFException:
            break

    if vis:
        return

    category_iou, avg_iou = conf_mat.mean_iou()
    category_acc, avg_acc = conf_mat.accuracy()
    fwiou = conf_mat.frequency_weighted_iou()
    print("[EVAL]#image={} acc={:.4f} IoU={:.4f} FWIoU={:.4f}".format(
        num_images, avg_acc, avg_iou, fwiou))
    print("[EVAL]Category Acc:", category_acc)
    print("[EVAL]Category IoU:", category_iou)
    print("[EVAL]Kappa: {:.4f}".format(conf_mat.kappa()))

    return category_iou, avg_iou, category_acc, avg_acc


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

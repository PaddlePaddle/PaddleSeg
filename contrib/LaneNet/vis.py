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
SEG_PATH = os.path.join(cur_path, "../../../")
sys.path.append(SEG_PATH)
sys.path.append(root_path)

import matplotlib
matplotlib.use('Agg')
import time
import argparse
import pprint
import cv2
import numpy as np
import paddle.fluid as fluid

from utils.config import cfg
from reader import LaneNetDataset
from models.model_builder import build_model
from models.model_builder import ModelPhase
from utils import lanenet_postprocess
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='PaddeSeg visualization tools')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str)
    parser.add_argument(
        '--use_gpu', dest='use_gpu', help='Use gpu or cpu', action='store_true')
    parser.add_argument(
        '--vis_dir',
        dest='vis_dir',
        help='visual save dir',
        type=str,
        default='visual')
    parser.add_argument(
        '--also_save_raw_results',
        dest='also_save_raw_results',
        help='whether to save raw result',
        action='store_true')
    parser.add_argument(
        '--local_test',
        dest='local_test',
        help='if in local test mode, only visualize 5 images for testing',
        action='store_true')
    parser.add_argument(
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def to_png_fn(fn, name=""):
    """
    Append png as filename postfix
    """
    directory, filename = os.path.split(fn)
    basename, ext = os.path.splitext(filename)

    return basename + name + ".png"


def minmax_scale(input_arr):
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def visualize(cfg,
              vis_file_list=None,
              use_gpu=False,
              vis_dir="visual",
              also_save_raw_results=False,
              ckpt_dir=None,
              log_writer=None,
              local_test=False,
              **kwargs):
    if vis_file_list is None:
        vis_file_list = cfg.DATASET.TEST_FILE_LIST

    dataset = LaneNetDataset(
        file_list=vis_file_list,
        mode=ModelPhase.VISUAL,
        shuffle=True,
        data_dir=cfg.DATASET.DATA_DIR)

    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    pred, logit = build_model(test_prog, startup_prog, phase=ModelPhase.VISUAL)
    # Clone forward graph
    test_prog = test_prog.clone(for_test=True)

    # Get device environment
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    ckpt_dir = cfg.TEST.TEST_MODEL if not ckpt_dir else ckpt_dir

    if ckpt_dir is not None:
        print('load test model:', ckpt_dir)
        try:
            fluid.load(test_prog, os.path.join(ckpt_dir, 'model'), exe)
        except:
            fluid.io.load_params(exe, ckpt_dir, main_program=test_prog)

    save_dir = os.path.join(vis_dir, 'visual_results')
    makedirs(save_dir)
    if also_save_raw_results:
        raw_save_dir = os.path.join(vis_dir, 'raw_results')
        makedirs(raw_save_dir)

    fetch_list = [pred.name, logit.name]
    test_reader = dataset.batch(dataset.generator, batch_size=1, is_test=True)

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()
    for imgs, grts, grts_instance, img_names, valid_shapes, org_imgs in test_reader:
        segLogits, emLogits = exe.run(
            program=test_prog,
            feed={'image': imgs},
            fetch_list=fetch_list,
            return_numpy=True)
        num_imgs = segLogits.shape[0]

        for i in range(num_imgs):
            gt_image = org_imgs[i]
            binary_seg_image, instance_seg_image = segLogits[i].squeeze(
                -1), emLogits[i].transpose((1, 2, 0))

            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image,
                instance_seg_result=instance_seg_image,
                source_image=gt_image)
            pred_binary_fn = os.path.join(
                save_dir, to_png_fn(img_names[i], name='_pred_binary'))
            pred_lane_fn = os.path.join(
                save_dir, to_png_fn(img_names[i], name='_pred_lane'))
            pred_instance_fn = os.path.join(
                save_dir, to_png_fn(img_names[i], name='_pred_instance'))
            dirname = os.path.dirname(pred_binary_fn)

            makedirs(dirname)
            mask_image = postprocess_result['mask_image']
            for i in range(4):
                instance_seg_image[:, :, i] = minmax_scale(
                    instance_seg_image[:, :, i])
            embedding_image = np.array(instance_seg_image).astype(np.uint8)

            plt.figure('mask_image')
            plt.imshow(mask_image[:, :, (2, 1, 0)])
            plt.figure('src_image')
            plt.imshow(gt_image[:, :, (2, 1, 0)])
            plt.figure('instance_image')
            plt.imshow(embedding_image[:, :, (2, 1, 0)])
            plt.figure('binary_image')
            plt.imshow(binary_seg_image * 255, cmap='gray')
            plt.show()

            cv2.imwrite(pred_binary_fn,
                        np.array(binary_seg_image * 255).astype(np.uint8))
            cv2.imwrite(pred_lane_fn, postprocess_result['source_image'])
            cv2.imwrite(pred_instance_fn, mask_image)
            print(pred_lane_fn, 'saved!')


if __name__ == '__main__':
    args = parse_args()
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)
    cfg.check_and_infer()
    print(pprint.pformat(cfg))
    visualize(cfg, **args.__dict__)

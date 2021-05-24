# coding: utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import os.path as osp

import cv2
import numpy as np
from paddleseg.utils import get_sys_env, logger

from deploy.infer import Predictor


def parse_args():
    parser = argparse.ArgumentParser(description='HumanSeg inference for video')
    parser.add_argument(
        "--cfg",
        dest="cfg",
        help="The config file.",
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        "--input_shape",
        dest="input_shape",
        help="The image shape for net inputs.",
        nargs=2,
        default=[192, 192],
        type=int)
    parser.add_argument(
        '--video_path',
        dest='video_path',
        help=
        'Video path for inference, camera will be used if the path not existing',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./output')

    parser.add_argument(
        '--with_argmax',
        dest='with_argmax',
        help='Perform argmax operation on the predict result.',
        action='store_true')
    parser.add_argument(
        '--not_soft_predict',
        dest='not_soft_predict',
        help='',
        action='store_true')

    return parser.parse_args()


def video_infer(args):
    env_info = get_sys_env()
    args.use_gpu = True if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else False
    predictor = Predictor(args)

    if not args.video_path:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise IOError("Error opening video stream or file, "
                      "Please check `--video_path`:{} exists"
                      " or the camera is working properly".format(
                          args.video_path))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.video_path:
        logger.info('Please wait. It is computing......')
        # 用于保存预测结果视频
        if not osp.exists(args.save_dir):
            os.makedirs(args.save_dir)
        out = cv2.VideoWriter(
            osp.join(args.save_dir, 'result.avi'),
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
        # 开始获取视频帧
        frame = 0
        while cap.isOpened():
            ret, img = cap.read()
            if ret:
                bg = 255 * np.ones_like(img)
                comb = predictor.run(img, bg)
                out.write(comb)
                frame += 1
                logger.info('Processing frame {}'.format(frame))
            else:
                break
        cap.release()
        out.release()

    # 当没有输入预测图像和视频的时候，则打开摄像头
    else:
        while cap.isOpened():
            ret, img = cap.read()
            if ret:
                bg = 255 * np.ones_like(img)
                comb = predictor.run(img, bg)
                cv2.imshow('HumanSegmentation', comb)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()


if __name__ == "__main__":
    args = parse_args()
    video_infer(args)

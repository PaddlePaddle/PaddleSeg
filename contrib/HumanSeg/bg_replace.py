# coding: utf8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
        '--image_path',
        dest='image_path',
        help='Image including human',
        type=str,
        default=None)
    parser.add_argument(
        '--background_image_path',
        dest='background_image_path',
        help='Background image for replacing',
        type=str,
        default=None)
    parser.add_argument(
        '--video_path',
        dest='video_path',
        help='Video path for inference',
        type=str,
        default=None)
    parser.add_argument(
        '--background_video_path',
        dest='background_video_path',
        help='Background video path for replacing',
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


def background_replace(args):
    env_info = get_sys_env()
    args.use_gpu = True if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else False
    predictor = Predictor(args)

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 图像背景替换
    if args.image_path is not None:
        if not osp.exists(args.image_path):
            raise Exception('The --image_path is not existed: {}'.format(
                args.image_path))
        if args.background_image_path is None:
            raise Exception(
                'The --background_image_path is not set. Please set it')
        else:
            if not osp.exists(args.background_image_path):
                raise Exception(
                    'The --background_image_path is not existed: {}'.format(
                        args.background_image_path))
        img = cv2.imread(args.image_path)
        bg = cv2.imread(args.background_image_path)

        comb = predictor.run(img, bg)

        save_name = osp.basename(args.image_path)
        save_path = osp.join(args.save_dir, save_name)
        cv2.imwrite(save_path, comb)

    # 视频背景替换
    else:
        # 如果提供背景视频则以背景视频作为背景，否则采用提供的背景图片
        is_video_bg = False
        if args.background_video_path is not None:
            if not osp.exists(args.background_video_path):
                raise Exception(
                    'The --background_video_path is not existed: {}'.format(
                        args.background_video_path))
            is_video_bg = True
        elif args.background_image_path is not None:
            if not osp.exists(args.background_image_path):
                raise Exception(
                    'The --background_image_path is not existed: {}'.format(
                        args.background_image_path))
        else:
            raise Exception(
                'Please offer backgound image or video. You should set --backbground_iamge_paht or --background_video_path'
            )

        if args.video_path is not None:
            logger.info('Please wait. It is computing......')
            if not osp.exists(args.video_path):
                raise Exception('The --video_path is not existed: {}'.format(
                    args.video_path))

            cap_video = cv2.VideoCapture(args.video_path)
            fps = cap_video.get(cv2.CAP_PROP_FPS)
            width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            save_name = osp.basename(args.video_path)
            save_name = save_name.split('.')[0]
            save_path = osp.join(args.save_dir, save_name + '.avi')

            cap_out = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                (width, height))

            if is_video_bg:
                cap_bg = cv2.VideoCapture(args.background_video_path)
                frames_bg = cap_bg.get(cv2.CAP_PROP_FRAME_COUNT)
                current_frame_bg = 1
            else:
                img_bg = cv2.imread(args.background_image_path)
            frame_num = 0
            while cap_video.isOpened():
                ret, frame = cap_video.read()
                if ret:
                    #循环读取背景帧
                    if is_video_bg:
                        ret_bg, frame_bg = cap_bg.read()
                        if ret_bg:
                            if current_frame_bg == frames_bg:
                                current_frame_bg = 1
                                cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        else:
                            break
                        current_frame_bg += 1
                        comb = predictor.run(frame, frame_bg)
                    else:
                        comb = predictor.run(frame, img_bg)

                    cap_out.write(comb)
                    frame_num += 1
                    logger.info('Processing frame {}'.format(frame_num))
                else:
                    break

            if is_video_bg:
                cap_bg.release()
            cap_video.release()
            cap_out.release()

        # 当没有输入预测图像和视频的时候，则打开摄像头
        else:
            cap_video = cv2.VideoCapture(0)
            if not cap_video.isOpened():
                raise IOError("Error opening video stream or file, "
                              "--video_path whether existing: {}"
                              " or camera whether working".format(
                                  args.video_path))
                return

            if is_video_bg:
                cap_bg = cv2.VideoCapture(args.background_video_path)
                frames_bg = cap_bg.get(cv2.CAP_PROP_FRAME_COUNT)
                current_frame_bg = 1
            else:
                img_bg = cv2.imread(args.background_image_path)

            while cap_video.isOpened():
                ret, frame = cap_video.read()
                if ret:
                    #循环读取背景帧
                    if is_video_bg:
                        ret_bg, frame_bg = cap_bg.read()
                        if ret_bg:
                            if current_frame_bg == frames_bg:
                                current_frame_bg = 1
                                cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        else:
                            break
                        current_frame_bg += 1
                        comb = predictor.run(frame, frame_bg)
                    else:
                        comb = predictor.run(frame, img_bg)
                    cv2.imshow('HumanSegmentation', comb)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            if is_video_bg:
                cap_bg.release()
            cap_video.release()


if __name__ == "__main__":
    args = parse_args()
    background_replace(args)

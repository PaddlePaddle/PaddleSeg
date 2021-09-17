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
    parser = argparse.ArgumentParser(description='PP-HumanSeg inference for video')
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        "--input_shape",
        dest="input_shape",
        help="The image shape [h, w] for net inputs.",
        nargs=2,
        default=[192, 192],
        type=int)
    parser.add_argument(
        '--img_path',
        dest='img_path',
        help='Image including human',
        type=str,
        default=None)
    parser.add_argument(
        '--video_path',
        dest='video_path',
        help='Video path for inference',
        type=str,
        default=None)
    parser.add_argument(
        '--bg_img_path',
        dest='bg_img_path',
        help=
        'Background image path for replacing. If not specified, a white background is used',
        type=str,
        default=None)
    parser.add_argument(
        '--bg_video_path',
        dest='bg_video_path',
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
        help=
        'If this is turned on, the prediction result will be output directly without using soft predict',
        action='store_true')

    parser.add_argument(
        '--test_speed',
        dest='test_speed',
        help='Whether to test inference speed',
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
    if args.img_path is not None:
        if not osp.exists(args.img_path):
            raise Exception('The --img_path is not existed: {}'.format(
                args.img_path))
        img = cv2.imread(args.img_path)
        bg = get_bg_img(args.bg_img_path, img.shape)

        comb = predictor.run(img, bg)

        save_name = osp.basename(args.img_path)
        save_path = osp.join(args.save_dir, save_name)
        cv2.imwrite(save_path, comb)
    # 视频背景替换
    else:
        # 获取背景：如果提供背景视频则以背景视频作为背景，否则采用提供的背景图片
        if args.bg_video_path is not None:
            if not osp.exists(args.bg_video_path):
                raise Exception('The --bg_video_path is not existed: {}'.format(
                    args.bg_video_path))
            is_video_bg = True
        else:
            bg = get_bg_img(args.bg_img_path, args.input_shape)
            is_video_bg = False

        # 视频预测
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
                cap_bg = cv2.VideoCapture(args.bg_video_path)
                frames_bg = cap_bg.get(cv2.CAP_PROP_FRAME_COUNT)
                current_bg = 1
            frame_num = 0
            while cap_video.isOpened():
                ret, frame = cap_video.read()
                if ret:
                    #读取背景帧
                    if is_video_bg:
                        ret_bg, bg = cap_bg.read()
                        if ret_bg:
                            if current_bg == frames_bg:
                                current_bg = 1
                                cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        else:
                            break
                        current_bg += 1

                    comb = predictor.run(frame, bg)

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
                cap_bg = cv2.VideoCapture(args.bg_video_path)
                frames_bg = cap_bg.get(cv2.CAP_PROP_FRAME_COUNT)
                current_bg = 1

            while cap_video.isOpened():
                ret, frame = cap_video.read()
                if ret:
                    #读取背景帧
                    if is_video_bg:
                        ret_bg, bg = cap_bg.read()
                        if ret_bg:
                            if current_bg == frames_bg:
                                current_bg = 1
                                cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        else:
                            break
                        current_bg += 1

                    comb = predictor.run(frame, bg)

                    cv2.imshow('HumanSegmentation', comb)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            if is_video_bg:
                cap_bg.release()
            cap_video.release()
    if args.test_speed:
        timer = predictor.cost_averager
        logger.info(
            'Model inference time per image: {}\nFPS: {}\nNum of images: {}'.
            format(timer.get_average(), 1 / timer.get_average(), timer._cnt))


def get_bg_img(bg_img_path, img_shape):
    if bg_img_path is None:
        bg = 255 * np.ones(img_shape)
    elif not osp.exists(bg_img_path):
        raise Exception(
            'The --bg_img_path is not existed: {}'.format(bg_img_path))
    else:
        bg = cv2.imread(bg_img_path)
    return bg


if __name__ == "__main__":
    args = parse_args()
    background_replace(args)

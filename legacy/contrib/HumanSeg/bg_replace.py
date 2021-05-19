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

from utils.humanseg_postprocess import postprocess, threshold_mask
import models
import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='HumanSeg inference for video')
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='Model path for inference',
        type=str)
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
        "--image_shape",
        dest="image_shape",
        help="The image shape for net inputs.",
        nargs=2,
        default=[192, 192],
        type=int)

    return parser.parse_args()


def predict(img, model, test_transforms):
    model.arrange_transform(transforms=test_transforms, mode='test')
    img, im_info = test_transforms(img)
    img = np.expand_dims(img, axis=0)
    result = model.exe.run(
        model.test_prog,
        feed={'image': img},
        fetch_list=list(model.test_outputs.values()))
    score_map = result[1]
    score_map = np.squeeze(score_map, axis=0)
    score_map = np.transpose(score_map, (1, 2, 0))
    return score_map, im_info


def recover(img, im_info):
    keys = list(im_info.keys())
    for k in keys[::-1]:
        if k == 'shape_before_resize':
            h, w = im_info[k][0], im_info[k][1]
            img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
        elif k == 'shape_before_padding':
            h, w = im_info[k][0], im_info[k][1]
            img = img[0:h, 0:w]
    return img


def bg_replace(score_map, img, bg):
    h, w, _ = img.shape
    bg = cv2.resize(bg, (w, h))
    score_map = np.repeat(score_map[:, :, np.newaxis], 3, axis=2)
    comb = (score_map * img + (1 - score_map) * bg).astype(np.uint8)
    return comb


def infer(args):
    resize_h = args.image_shape[1]
    resize_w = args.image_shape[0]

    test_transforms = transforms.Compose(
        [transforms.Resize((resize_w, resize_h)),
         transforms.Normalize()])
    model = models.load_model(args.model_dir)

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
        score_map, im_info = predict(img, model, test_transforms)
        score_map = score_map[:, :, 1]
        score_map = recover(score_map, im_info)
        bg = cv2.imread(args.background_image_path)
        save_name = osp.basename(args.image_path)
        save_path = osp.join(args.save_dir, save_name)
        result = bg_replace(score_map, img, bg)
        cv2.imwrite(save_path, result)

    # 视频背景替换，如果提供背景视频则以背景视频作为背景，否则采用提供的背景图片
    else:
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

        disflow = cv2.DISOpticalFlow_create(
            cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        prev_gray = np.zeros((resize_h, resize_w), np.uint8)
        prev_cfd = np.zeros((resize_h, resize_w), np.float32)
        is_init = True
        if args.video_path is not None:
            print('Please wait. It is computing......')
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
            while cap_video.isOpened():
                ret, frame = cap_video.read()
                if ret:
                    score_map, im_info = predict(frame, model, test_transforms)
                    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cur_gray = cv2.resize(cur_gray, (resize_w, resize_h))
                    score_map = 255 * score_map[:, :, 1]
                    optflow_map = postprocess(cur_gray, score_map, prev_gray, prev_cfd, \
                                              disflow, is_init)
                    prev_gray = cur_gray.copy()
                    prev_cfd = optflow_map.copy()
                    is_init = False
                    optflow_map = cv2.GaussianBlur(optflow_map, (3, 3), 0)
                    optflow_map = threshold_mask(
                        optflow_map, thresh_bg=0.2, thresh_fg=0.8)
                    score_map = recover(optflow_map, im_info)

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
                        comb = bg_replace(score_map, frame, frame_bg)
                    else:
                        comb = bg_replace(score_map, frame, img_bg)

                    cap_out.write(comb)
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
                    score_map, im_info = predict(frame, model, test_transforms)
                    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cur_gray = cv2.resize(cur_gray, (resize_w, resize_h))
                    score_map = 255 * score_map[:, :, 1]
                    optflow_map = postprocess(cur_gray, score_map, prev_gray, prev_cfd, \
                                              disflow, is_init)
                    prev_gray = cur_gray.copy()
                    prev_cfd = optflow_map.copy()
                    is_init = False
                    optflow_map = cv2.GaussianBlur(optflow_map, (3, 3), 0)
                    optflow_map = threshold_mask(
                        optflow_map, thresh_bg=0.2, thresh_fg=0.8)
                    score_map = recover(optflow_map, im_info)

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
                        comb = bg_replace(score_map, frame, frame_bg)
                    else:
                        comb = bg_replace(score_map, frame, img_bg)
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
    infer(args)

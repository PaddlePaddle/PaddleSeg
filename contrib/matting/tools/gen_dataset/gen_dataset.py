# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
to generator the human matting dataset. The directory is as follow:
     human_matting
          train
               image
                    000001.png
                    ...
               fg
                    000001.png
                    ...
               bg
                    000001.png
                    ...
               alpha
                    000001.png
                    ....
               trimap
                    000001.png
          val
               image
                    000001.png
                    ...
               fg
                    000001.png
                    ...
               bg
                    000001.png
                    ...
               alpha
                    000001.png
                    ....
               trimap
                    000001.png
                    ...
For video, get one every 5 frames, and composite it with one background.
For image, one image is composited with 5 background.
"""

import os
import math
import time

import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


def get_files(root_path):
    res = []
    for root, dirs, files in os.walk(root_path, followlinks=True):
        for f in files:
            if f.endswith(('.jpg', '.png', '.jpeg', 'JPG', '.mp4')):
                res.append(os.path.join(root, f))
    return res


ori_dataset_root = "/mnt/chenguowei01/datasets/matting/gather"
ori_fg_path = os.path.join(ori_dataset_root, 'fg')
ori_alpha_path = os.path.join(ori_dataset_root, 'alpha')
ori_bg_path = os.path.join(ori_dataset_root, 'bg')

fg_list = get_files(ori_fg_path)
alpha_list = [f.replace('fg', 'alpha') for f in fg_list]
bg_list = get_files(ori_bg_path)
len_bg_list = len(bg_list)

dataset_root = '/ssd3/chenguowei01/datasets/matting/human_matting'


def im_write(save_path, img):
    dir_name = os.path.dirname(save_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cv2.imwrite(save_path, img)


def composite(fg, alpha, ori_bg):
    fg_h, fg_w = fg.shape[:2]
    ori_bg_h, ori_bg_w = ori_bg.shape[:2]

    wratio = fg_w / ori_bg_w
    hratio = fg_h / ori_bg_h
    ratio = wratio if wratio > hratio else hratio

    # Resize ori_bg if it is smaller than fg.
    if ratio > 1:
        resize_h = math.ceil(ori_bg_h * ratio)
        resize_w = math.ceil(ori_bg_w * ratio)
        bg = cv2.resize(
            ori_bg, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    else:
        bg = ori_bg

    bg = bg[0:fg_h, 0:fg_w, :]
    alpha = alpha / 255
    alpha = np.expand_dims(alpha, axis=2)
    image = alpha * fg + (1 - alpha) * bg
    image = image.astype(np.uint8)
    return image, bg


def video_comp(fg_file, alpha_file, bg_index_list, interval=5, mode='train'):
    fg_video_capture = cv2.VideoCapture(fg_file)
    alpha_video_capture = cv2.VideoCapture(alpha_file)
    frames = fg_video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("there are {} frames in video {}".format(frames, fg_file))

    f_index = 0
    while True:
        if f_index >= frames:
            break
        fg_video_capture.set(cv2.CAP_PROP_POS_FRAMES, f_index)
        fg_ret, fg_frame = fg_video_capture.retrieve()  # get foreground
        alpha_video_capture.set(cv2.CAP_PROP_POS_FRAMES, f_index)
        alpha_ret, alpha_frame = alpha_video_capture.retrieve()  # get alpha
        ret = fg_ret and alpha_ret
        if not ret:
            break
        if len(alpha_frame.shape) == 3:
            alpha_frame = alpha_frame[:, :, 0]

        file_name = os.path.basename(fg_file)
        file_name = os.path.splitext(file_name)[0]
        fg_save_name = os.path.join(dataset_root, mode, 'fg', file_name,
                                    '{:0>5d}'.format(f_index) + '.png')
        alpha_save_name = fg_save_name.replace('fg', 'alpha')
        bg_save_name = fg_save_name.replace('fg', 'bg')
        image_save_name = fg_save_name.replace('fg', 'image')

        ori_bg = cv2.imread(
            bg_list[bg_index_list[f_index % len_bg_list]])  # get background
        image, bg = composite(
            fg_frame, alpha_frame,
            ori_bg)  # get composition image and the response background

        # save fg, alpha, bg, image
        im_write(fg_save_name, fg_frame)
        im_write(alpha_save_name, alpha_frame)
        im_write(image_save_name, image)
        im_write(bg_save_name, bg)

        f_index += interval


def image_comp(fg_file, alpha_file, bg_index_list, num_bgs=5, mode='train'):
    fg = cv2.imread(fg_file)
    alpha = cv2.imread(alpha_file, cv2.IMREAD_UNCHANGED)
    print('Composition for ', fg_file)

    for i in range(num_bgs):
        bg_index = bg_index_list[i]
        ori_bg = cv2.imread(bg_list[bg_index])  # get background
        image, bg = composite(fg, alpha, ori_bg)

        file_name = os.path.basename(fg_file)
        file_name = os.path.splitext(file_name)[0]
        file_name = '_'.join([file_name, '{:0>3d}'.format(i)])
        fg_save_name = os.path.join(dataset_root, mode, 'fg',
                                    file_name + '.png')
        alpha_save_name = fg_save_name.replace('fg', 'alpha')
        bg_save_name = fg_save_name.replace('fg', 'bg')
        image_save_name = fg_save_name.replace('fg', 'image')

        im_write(fg_save_name, fg)
        im_write(alpha_save_name, alpha)
        im_write(image_save_name, image)
        im_write(bg_save_name, bg)


def comp_one(fa_index):
    """
     Composite foreground and background.

     Args:
          fa_index: The index of foreground and alpha.
          bg_index: The index of background, if foreground is video, get one every 5 frames, and composite it with one background，
               if foreground is image, one image is composited with 5 background.
     """
    fg_file = fg_list[fa_index]
    alpha_file = alpha_list[fa_index]
    mode = 'train' if 'train' in fg_file else 'val'

    # Randomly bg index
    np.random.seed(int(os.getpid() * time.time()) %
                   (2**30))  # make different for each process

    len_bg = len(bg_list)
    bg_index_list = list(range(len_bg))
    np.random.shuffle(bg_index_list)

    if os.path.splitext(fg_file)[-1] in ['.mp4']:
        video_comp(
            fg_file=fg_file,
            alpha_file=alpha_file,
            bg_index_list=bg_index_list,
            mode=mode)
    # else:
    #      image_comp(fg_file=fg_file, alpha_file=alpha_file, bg_index_list=bg_index_list, mode=mode)


def comp_pool():
    len_fa = len(fg_list)

    with Pool(20) as pool:
        with tqdm(total=len_fa) as pbar:
            for i, _ in tqdm(
                    enumerate(pool.imap_unordered(comp_one, range(len_fa)))):
                pbar.update()


if __name__ == '__main__':
    comp_pool()

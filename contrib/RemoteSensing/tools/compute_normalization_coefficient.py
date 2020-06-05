# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import os
import os.path as osp
import sys
import argparse
from tqdm import tqdm
import pickle
from data_analyse_and_check import read_img


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Compute normalization coefficient and clip percentage before training.'
    )
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='Dataset directory',
        default=None,
        type=str)
    parser.add_argument(
        '--pkl_path',
        dest='pkl_path',
        help='Path of img_pixel_statistics.pkl',
        default=None,
        type=str)
    parser.add_argument(
        '--clip_min_value',
        dest='clip_min_value',
        help='Min values for clipping data',
        nargs='+',
        default=None,
        type=int)
    parser.add_argument(
        '--clip_max_value',
        dest='clip_max_value',
        help='Max values for clipping data',
        nargs='+',
        default=None,
        type=int)
    parser.add_argument(
        '--separator',
        dest='separator',
        help='file list separator',
        default=" ",
        type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def compute_single_img(img):
    global MEANS, STDS, TOTAL_IMG_NUM, CLIP_MIN_VALUE, CLIP_MAX_VALUE

    TOTAL_IMG_NUM += 1
    channel = img.shape[2]
    if MEANS == []:
        MEANS = [0] * channel
    if STDS == []:
        STDS = [0] * channel
    for k in range(channel):
        if CLIP_MAX_VALUE != [] and CLIP_MIN_VALUE != []:
            np.clip(
                img[:, :, k],
                CLIP_MIN_VALUE[k],
                CLIP_MAX_VALUE[k],
                out=img[:, :, k])

            # Rescaling (min-max normalization)
            range_value = [
                CLIP_MAX_VALUE[i] - CLIP_MIN_VALUE[i]
                for i in range(len(CLIP_MAX_VALUE))
            ]
            img_k = (img[:, :, k].astype(np.float32, copy=False) -
                     CLIP_MIN_VALUE[k]) / range_value[k]
        else:
            img_k = img[:, :, k]

        # count mean, std
        img_mean = np.mean(img_k)
        img_std = np.std(img_k)
        MEANS[k] += img_mean
        STDS[k] += img_std


def compute_normalize_coefficient():
    global MEANS, STDS
    for file_list in [TRAIN_FILE_LIST, VAL_FILE_LIST, TEST_FILE_LIST]:
        with open(file_list, 'r') as fid:
            lines = fid.readlines()
            if not lines:
                print("File list is empty!")
                continue
            for line in tqdm(lines):
                line = line.strip()
                parts = line.split(SEPARATOR)
                img_name, grt_name = parts[0], parts[1]
                img_path = os.path.join(DATA_DIR, img_name)
                img = read_img(img_path)

                compute_single_img(img)

    # count mean, std
    MEANS = [i / TOTAL_IMG_NUM for i in MEANS]
    STDS = [i / TOTAL_IMG_NUM for i in STDS]
    print("\nCount the channel-by-channel mean and std of the image:\n"
          "mean = {}\nstd = {}".format(MEANS, STDS))


def compute_clip_percentage():
    with open(PKL_PATH, 'rb') as f:
        percentage, img_value_num = pickle.load(f)

    for k in range(len(img_value_num)):
        # count clip value percentage
        range_pixel = 0
        for i, element in enumerate(img_value_num[k]):
            if CLIP_MIN_VALUE[k] <= i <= CLIP_MAX_VALUE[k]:
                range_pixel += element
        sum_pixel = sum(img_value_num[k])
        print('channel {}, clip value percentage = {}'.format(
            k, range_pixel / sum_pixel))


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR = args.data_dir
    TRAIN_FILE_LIST = osp.join(DATA_DIR, 'train.txt')
    VAL_FILE_LIST = osp.join(DATA_DIR, 'val.txt')
    TEST_FILE_LIST = osp.join(DATA_DIR, 'test.txt')
    SEPARATOR = args.separator
    CLIP_MIN_VALUE = args.clip_min_value
    CLIP_MAX_VALUE = args.clip_max_value
    PKL_PATH = args.pkl_path
    MEANS = []
    STDS = []
    TOTAL_IMG_NUM = 0

    compute_normalize_coefficient()
    compute_clip_percentage()

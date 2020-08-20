# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


def compute_single_img(img, clip_min_value, clip_max_value):
    channel = img.shape[2]
    means = np.zeros(channel)
    stds = np.zeros(channel)
    for k in range(channel):
        if clip_max_value != [] and clip_min_value != []:
            np.clip(
                img[:, :, k],
                clip_min_value[k],
                clip_max_value[k],
                out=img[:, :, k])

            # Rescaling (min-max normalization)
            range_value = [
                clip_max_value[i] - clip_min_value[i]
                for i in range(len(clip_max_value))
            ]
            img_k = (img[:, :, k].astype(np.float32, copy=False) -
                     clip_min_value[k]) / range_value[k]
        else:
            img_k = img[:, :, k]

        # count mean, std
        means[k] = np.mean(img_k)
        stds[k] = np.std(img_k)
    return means, stds


def cal_normalize_coefficient(data_dir, separator, clip_min_value,
                              clip_max_value):
    train_file_list = osp.join(data_dir, 'train.txt')
    val_file_list = osp.join(data_dir, 'val.txt')
    test_file_list = osp.join(data_dir, 'test.txt')
    total_img_num = 0
    for file_list in [train_file_list, val_file_list, test_file_list]:
        with open(file_list, 'r') as fid:
            print("\n-----------------------------\nCheck {}...".format(
                file_list))
            lines = fid.readlines()
            if not lines:
                print("File list is empty!")
                continue
            for line in tqdm(lines):
                line = line.strip()
                parts = line.split(separator)
                img_name, grt_name = parts[0], parts[1]
                img_path = os.path.join(data_dir, img_name)
                img = read_img(img_path)
                if total_img_num == 0:
                    channel = img.shape[2]
                    total_means = np.zeros(channel)
                    total_stds = np.zeros(channel)
                means, stds = compute_single_img(img, clip_min_value,
                                                 clip_max_value)
                total_means += means
                total_stds += stds
                total_img_num += 1

    # count mean, std
    total_means = total_means / total_img_num
    total_stds = total_stds / total_img_num
    print("\nCount the channel-by-channel mean and std of the image:\n"
          "mean = {}\nstd = {}".format(total_means, total_stds))


def cal_clip_percentage(pkl_path, clip_min_value, clip_max_value):
    """
    Calculate the percentage of pixels to be clipped
    """
    with open(pkl_path, 'rb') as f:
        percentage, img_value_num = pickle.load(f)

    for k in range(len(img_value_num)):
        range_pixel = 0
        for i, element in enumerate(img_value_num[k]):
            if clip_min_value[k] <= i <= clip_max_value[k]:
                range_pixel += element
        sum_pixel = sum(img_value_num[k])
        print('channel {}, the percentage of pixels to be clipped = {}'.format(
            k, 1 - range_pixel / sum_pixel))


def main():
    args = parse_args()
    data_dir = args.data_dir
    separator = args.separator
    clip_min_value = args.clip_min_value
    clip_max_value = args.clip_max_value
    pkl_path = args.pkl_path

    cal_normalize_coefficient(data_dir, separator, clip_min_value,
                              clip_max_value)
    cal_clip_percentage(pkl_path, clip_min_value, clip_max_value)


if __name__ == "__main__":
    main()

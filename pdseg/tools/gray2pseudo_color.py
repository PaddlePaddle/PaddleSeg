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

from __future__ import print_function

import argparse
import os
import os.path as osp
import sys
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dir_or_file', help='input gray label directory or file list path')
    parser.add_argument('output_dir', help='output colorful label directory')
    parser.add_argument('--dataset_dir', help='dataset directory')
    parser.add_argument('--file_separator', help='file list separator')
    return parser.parse_args()


def get_color_map_list(num_classes):
    """ Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3

    return color_map


def gray2pseudo_color(args):
    """将灰度标注图片转换为伪彩色图片"""
    input = args.dir_or_file
    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        print('Creating colorful label directory:', output_dir)

    color_map = get_color_map_list(256)
    if os.path.isdir(input):
        for fpath, dirs, fs in os.walk(input):
            for f in fs:
                try:
                    grt_path = osp.join(fpath, f)
                    _output_dir = fpath.replace(input, '')
                    _output_dir = _output_dir.lstrip(os.path.sep)

                    im = Image.open(grt_path)
                    lbl = np.asarray(im)

                    lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode='P')
                    lbl_pil.putpalette(color_map)

                    real_dir = osp.join(output_dir, _output_dir)
                    if not osp.exists(real_dir):
                        os.makedirs(real_dir)
                    new_grt_path = osp.join(real_dir, f)

                    lbl_pil.save(new_grt_path)
                    print('New label path:', new_grt_path)
                except:
                    continue
    elif os.path.isfile(input):
        if args.dataset_dir is None or args.file_separator is None:
            print('No dataset_dir or file_separator input!')
            sys.exit()
        with open(input) as f:
            for line in f:
                parts = line.strip().split(args.file_separator)
                grt_name = parts[1]
                grt_path = os.path.join(args.dataset_dir, grt_name)

                im = Image.open(grt_path)
                lbl = np.asarray(im)

                lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode='P')
                lbl_pil.putpalette(color_map)

                grt_dir, _ = osp.split(grt_name)
                new_dir = osp.join(output_dir, grt_dir)
                if not osp.exists(new_dir):
                    os.makedirs(new_dir)
                new_grt_path = osp.join(output_dir, grt_name)

                lbl_pil.save(new_grt_path)
                print('New label path:', new_grt_path)
    else:
        print('It\'s neither a dir nor a file')


if __name__ == '__main__':
    args = parse_args()
    gray2pseudo_color(args)

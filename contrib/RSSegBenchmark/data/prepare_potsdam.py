# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# Adapted from https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/potsdam.py
#
# Original copyright info:
# Copyright (c) OpenMMLab. All rights reserved.
#
# See original LICENSE at https://github.com/open-mmlab/mmsegmentation/blob/master/LICENSE

import argparse
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert potsdam dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='potsdam folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    args = parser.parse_args()
    return args


def clip_big_image(image_path, clip_save_dir, args, to_label=False):
    image = cv2.imread(image_path)

    h, w, c = image.shape
    clip_size = args.clip_size
    stride_size = args.stride_size

    num_rows = math.ceil((h - clip_size) / stride_size) if math.ceil(
        (h - clip_size
         ) / stride_size) * stride_size + clip_size >= h else math.ceil(
             (h - clip_size) / stride_size) + 1
    num_cols = math.ceil((w - clip_size) / stride_size) if math.ceil(
        (w - clip_size
         ) / stride_size) * stride_size + clip_size >= w else math.ceil(
             (w - clip_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * clip_size
    ymin = y * clip_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack(
        [
            xmin + xmin_offset, ymin + ymin_offset,
            np.minimum(xmin + clip_size, w), np.minimum(ymin + clip_size, h)
        ],
        axis=1)

    if to_label:
        color_map = np.array([[255, 255, 255], [255, 0, 0], [255, 255, 0],
                              [0, 255, 0], [0, 255, 255], [0, 0, 255],
                              [0, 0, 0]])
        flatten_v = np.matmul(
            image.reshape(-1, c), np.array([2, 3, 4]).reshape(3, 1))
        out = np.zeros_like(flatten_v)
        for idx, class_color in enumerate(color_map):
            value_idx = np.matmul(class_color,
                                  np.array([2, 3, 4]).reshape(3, 1))
            if idx == 6:
                out[flatten_v == value_idx] = 255
            else:
                out[flatten_v == value_idx] = idx
        image = out.reshape(h, w)

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y, start_x:
                              end_x] if to_label else image[start_y:end_y,
                                                            start_x:end_x, :]
        idx_i, idx_j = osp.basename(image_path).split('_')[2:4]
        cv2.imwrite(
            osp.join(
                clip_save_dir,
                f'{idx_i}_{idx_j}_{start_x}_{start_y}_{end_x}_{end_y}.png'),
            clipped_image.astype(np.uint8))


def main():
    args = parse_args()
    splits = {
        'train': [
            '2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11',
            '4_12', '5_10', '5_11', '5_12', '6_10', '6_11', '6_12', '6_7',
            '6_8', '6_9', '7_10', '7_11', '7_12', '7_7', '7_8', '7_9'
        ],
        'val': [
            '5_15', '6_15', '6_13', '3_13', '4_14', '6_14', '5_14', '2_13',
            '4_15', '2_14', '5_13', '4_13', '3_14', '7_13'
        ]
    }

    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'potsdam')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    if not osp.exists(osp.join(out_dir, 'img_dir', 'train')):
        os.makedirs(osp.join(out_dir, 'img_dir', 'train'))
    if not osp.exists(osp.join(out_dir, 'img_dir', 'val')):
        os.makedirs(osp.join(out_dir, 'img_dir', 'val'))
    if not osp.exists(osp.join(out_dir, 'ann_dir', 'train')):
        os.makedirs(osp.join(out_dir, 'ann_dir', 'train'))
    if not osp.exists(osp.join(out_dir, 'ann_dir', 'val')):
        os.makedirs(osp.join(out_dir, 'ann_dir', 'val'))

    zipp_list = glob.glob(os.path.join(dataset_path, '*.zip'))
    print('Find the data', zipp_list)

    for zipp in zipp_list:
        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
            zip_file = zipfile.ZipFile(zipp)
            zip_file.extractall(tmp_dir)
            src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
            if not len(src_path_list):
                sub_tmp_dir = os.path.join(tmp_dir, os.listdir(tmp_dir)[0])
                src_path_list = glob.glob(os.path.join(sub_tmp_dir, '*.tif'))

            for i, src_path in enumerate(src_path_list):
                idx_i, idx_j = osp.basename(src_path).split('_')[2:4]
                data_type = 'train' if f'{idx_i}_{idx_j}' in splits[
                    'train'] else 'val'
                if 'label' in src_path:
                    dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                    clip_big_image(src_path, dst_dir, args, to_label=True)
                else:
                    dst_dir = osp.join(out_dir, 'img_dir', data_type)
                    clip_big_image(src_path, dst_dir, args, to_label=False)

    print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    main()

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
# Adapted from https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/vaihingen.py
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
        description='Convert vaihingen dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='vaihingen folder path')
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


def clip_big_image(image_path, clip_save_dir, to_label=False):
    image = cv2.imread(image_path)

    h, w, c = image.shape
    cs = args.clip_size
    ss = args.stride_size

    num_rows = math.ceil((h - cs) / ss) if math.ceil(
        (h - cs) / ss) * ss + cs >= h else math.ceil((h - cs) / ss) + 1
    num_cols = math.ceil((w - cs) / ss) if math.ceil(
        (w - cs) / ss) * ss + cs >= w else math.ceil((w - cs) / ss) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * cs
    ymin = y * cs

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + cs > w, w - xmin - cs, np.zeros_like(xmin))
    ymin_offset = np.where(ymin + cs > h, h - ymin - cs, np.zeros_like(ymin))
    boxes = np.stack(
        [
            xmin + xmin_offset, ymin + ymin_offset, np.minimum(xmin + cs, w),
            np.minimum(ymin + cs, h)
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
        area_idx = osp.basename(image_path).split('_')[3].strip('.tif')
        cv2.imwrite(
            osp.join(clip_save_dir,
                     f'{area_idx}_{start_x}_{start_y}_{end_x}_{end_y}.png'),
            clipped_image.astype(np.uint8))


def main():
    splits = {
        'train': [
            'area1', 'area11', 'area13', 'area15', 'area17', 'area21', 'area23',
            'area26', 'area28', 'area3', 'area30', 'area32', 'area34', 'area37',
            'area5', 'area7'
        ],
        'val': [
            'area6', 'area24', 'area35', 'area16', 'area14', 'area22', 'area10',
            'area4', 'area2', 'area20', 'area8', 'area31', 'area33', 'area27',
            'area38', 'area12', 'area29'
        ],
    }

    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'vaihingen')
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

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for zipp in zipp_list:
            zip_file = zipfile.ZipFile(zipp)
            zip_file.extractall(tmp_dir)
            src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
            if 'ISPRS_semantic_labeling_Vaihingen' in zipp:
                src_path_list = glob.glob(
                    os.path.join(os.path.join(tmp_dir, 'top'), '*.tif'))
            if 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE' in zipp:  # noqa
                src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
                # delete unused area9 ground truth
                for area_ann in src_path_list:
                    if 'area9' in area_ann:
                        src_path_list.remove(area_ann)
            for i, src_path in enumerate(src_path_list):
                area_idx = osp.basename(src_path).split('_')[3].strip('.tif')
                data_type = 'train' if area_idx in splits['train'] else 'val'
                if 'noBoundary' in src_path:
                    dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                    clip_big_image(src_path, dst_dir, to_label=True)
                else:
                    dst_dir = osp.join(out_dir, 'img_dir', data_type)
                    clip_big_image(src_path, dst_dir, to_label=False)

        print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    main()

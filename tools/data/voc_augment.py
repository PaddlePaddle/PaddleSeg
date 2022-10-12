# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
File: voc_augment.py

This file use SBD(Semantic Boundaries Dataset) <http://home.bharathh.info/pubs/codes/SBD/download.html>
to augment the Pascal VOC.
"""

import os
import argparse
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from scipy.io import loadmat
import tqdm

from paddleseg.utils.download import download_file_and_uncompress

DATA_HOME = os.path.expanduser('~/.paddleseg/dataset/')
URL = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert SBD to Pascal Voc annotations to augment the train dataset of Pascal Voc'
    )
    parser.add_argument(
        '--voc_path',
        dest='voc_path',
        help='pascal voc path',
        type=str,
        default=os.path.join(DATA_HOME, 'VOCdevkit'))

    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='How many processes are used for data conversion',
        type=int,
        default=cpu_count())
    return parser.parse_args()


def mat_to_png(mat_file, sbd_cls_dir, save_dir):
    mat_path = os.path.join(sbd_cls_dir, mat_file)
    mat = loadmat(mat_path)
    mask = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
    save_file = os.path.join(save_dir, mat_file.replace('mat', 'png'))
    cv2.imwrite(save_file, mask)


def main():
    args = parse_args()
    sbd_path = download_file_and_uncompress(
        url=URL,
        savepath=DATA_HOME,
        extrapath=DATA_HOME,
        extraname='benchmark_RELEASE')
    with open(os.path.join(sbd_path, 'dataset/train.txt'), 'r') as f:
        sbd_file_list = [line.strip() for line in f]
    with open(os.path.join(sbd_path, 'dataset/val.txt'), 'r') as f:
        sbd_file_list += [line.strip() for line in f]
    if not os.path.exists(args.voc_path):
        raise FileNotFoundError(
            'There is no voc_path: {}. Please ensure that the Pascal VOC dataset has been downloaded correctly'
        )
    with open(
            os.path.join(args.voc_path,
                         'VOC2012/ImageSets/Segmentation/trainval.txt'),
            'r') as f:
        voc_file_list = [line.strip() for line in f]

    aug_file_list = list(set(sbd_file_list) - set(voc_file_list))
    with open(
            os.path.join(args.voc_path,
                         'VOC2012/ImageSets/Segmentation/aug.txt'), 'w') as f:
        f.writelines(''.join([line, '\n']) for line in aug_file_list)

    sbd_cls_dir = os.path.join(sbd_path, 'dataset/cls')
    save_dir = os.path.join(args.voc_path, 'VOC2012/SegmentationClassAug')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    mat_file_list = os.listdir(sbd_cls_dir)
    p = Pool(args.num_workers)
    for f in tqdm.tqdm(mat_file_list):
        p.apply_async(mat_to_png, args=(f, sbd_cls_dir, save_dir))
    p.close()
    p.join()


if __name__ == '__main__':
    main()

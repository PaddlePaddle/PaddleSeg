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

import os
import argparse
import shutil

import h5py
import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='prepare synapse')
    # params of training
    parser.add_argument(
        "--input_path",
        dest="input_path",
        help="The path of input files",
        default=None,
        type=str)

    parser.add_argument(
        '--output_path',
        dest='output_path',
        help='The path of output files',
        type=str,
        default=None)

    parser.add_argument(
        '--file_lists',
        dest='file_lists',
        help='The path of dataset split files',
        type=str,
        default=None)

    return parser.parse_args()


def main(args):

    sample_list = open(os.path.join(args.file_lists, 'train.txt')).readlines()
    train_dir = os.path.join(args.output_path, 'train')
    test_dir = os.path.join(args.output_path, 'test')
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir, ignore_errors=True)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)

    train_lines = []
    for sample in sample_list:
        sample = sample.strip('\n')
        data_path = os.path.join(args.input_path, 'train_npz', sample + '.npz')
        data = np.load(data_path)
        image, label = data['image'], data['label']

        np.save(os.path.join(train_dir, 'images', sample + '.npy'), image)
        np.save(os.path.join(train_dir, 'labels', sample + '.npy'), label)

        train_lines.append(
            os.path.join('train/images', sample + '.' + args.type) + " " +
            os.path.join('train/labels', sample + '.' + args.type) + "\n")
    with open(os.path.join(args.output_path, 'train.txt'), 'w+') as f:
        f.writelines(train_lines)

    test_lines = []
    sample_list = open(os.path.join(args.file_lists, 'test_vol.txt')).readlines(
    )
    for sample in sample_list:
        sample = sample.strip('\n')
        filepath = os.path.join(args.input_path, 'test_vol_h5',
                                "{}.npy.h5".format(sample))
        data = h5py.File(filepath)
        images, labels = data['image'][:], data['label'][:]
        filename = sample + '.npy'
        np.save(os.path.join(test_dir, 'images', filename), images)
        np.save(os.path.join(test_dir, 'labels', filename), labels)
        test_lines.append(
            os.path.join('test/images', filename) + " " + os.path.join(
                'test/labels', filename) + "\n")

        with open(os.path.join(args.output_path, 'test_list.txt'), 'w+') as f:
            f.writelines(test_lines)


if __name__ == '__main__':
    args = parse_args()
    main(args)

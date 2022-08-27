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
        '--type',
        dest='type',
        help='The directory for saving the model snapshot',
        type=str,
        default='npy')
    return parser.parse_args()


def get_color_map_list(num_classes):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.
    Args:
        num_classes (int): Number of classes.
    Returns:
        (list). The color map.
    """

    num_classes += 1
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
    color_map = color_map[3:]
    return color_map


def main(args):

    sample_list = open(
        os.path.join(args.input_path, 'lists/lists_Synapse',
                     'train.txt')).readlines()
    color_map = get_color_map_list(256)
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
        if args.type == 'png':
            image = image * 255.0
            lbl_pil = Image.fromarray(label.astype(np.uint8), mode='P')
            lbl_pil.putpalette(color_map)
            cv2.imwrite(
                os.path.join(train_dir, 'images', sample + '.png'), image)
            lbl_pil.save(os.path.join(train_dir, 'labels', sample + '.png'))
        elif args.type == 'npy':
            np.save(os.path.join(train_dir, 'images', sample + '.npy'), image)
            np.save(os.path.join(train_dir, 'labels', sample + '.npy'), label)
        else:
            raise NotImplementedError

        train_lines.append(
            os.path.join('train/images', sample + '.' + args.type) + " " +
            os.path.join('train/labels', sample + '.' + args.type) + "\n")
    with open(os.path.join(args.output_path, 'train.txt'), 'w+') as f:
        f.writelines(train_lines)

    test_lines = []
    sample_list = open(
        os.path.join(args.input_path, 'lists/lists_Synapse',
                     'test_vol.txt')).readlines()
    for sample in sample_list:
        sample = sample.strip('\n')
        filepath = os.path.join(args.input_path, 'test_vol_h5',
                                "{}.npy.h5".format(sample))
        data = h5py.File(filepath)
        images, labels = data['image'][:], data['label'][:]
        if args.type == 'png':
            for i in range(images.shape[0]):
                image = images[i]
                label = labels[i]
                image = image * 255.0
                label = label
                lbl_pil = Image.fromarray(label.astype(np.uint8), mode='P')
                lbl_pil.putpalette(color_map)
                filename = sample + f'_{i:>04d}.png'
                cv2.imwrite(os.path.join(test_dir, 'images', filename), image)
                lbl_pil.save(os.path.join(test_dir, 'labels', filename))
                test_lines.append(
                    os.path.join('test/images', filename) + " " + os.path.join(
                        'test/labels', filename) + "\n")
        elif args.type == 'npy':
            filename = sample + '.npy'
            np.save(os.path.join(test_dir, 'images', filename), images)
            np.save(os.path.join(test_dir, 'labels', filename), labels)
            test_lines.append(
                os.path.join('test/images', filename) + " " + os.path.join(
                    'test/labels', filename) + "\n")
        else:
            raise NotImplementedError

        with open(os.path.join(args.output_path, 'test.txt'), 'w+') as f:
            f.writelines(test_lines)


if __name__ == '__main__':
    args = parse_args()
    main(args)

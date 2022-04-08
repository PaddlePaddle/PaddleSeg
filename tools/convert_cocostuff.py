# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
File: convert_cocostuff.py
This file is based on https://github.com/nightrome/cocostuff to generate PASCAL-Context Dataset.
Before running, you should download the COCOSTUFF from https://github.com/nightrome/cocostuff. Then, make the folder
structure as follow:
cocostuff
|
|--images
|  |--train2017
|  |--val2017
|
|--annotations
|  |--train2017
|  |--val2017
"""

import os
import argparse

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Generate COCOStuff dataset')
    parser.add_argument(
        '--annotation_path',
        default='annotations',
        help='COCOStuff anotation path',
        type=str)
    parser.add_argument(
        '--save_path',
        default='convert_annotations',
        help='COCOStuff anotation path',
        type=str)

    return parser.parse_args()


class COCOStuffGenerator(object):
    def __init__(self, annotation_path, save_path):

        super(COCOStuffGenerator, self).__init__()

        self.mapping = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 66, 69, 71, 72, 73, 74, 75, 76, 77, 78,
            79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
            125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
            138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
            151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
            164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
            177, 178, 179, 180, 181
        ]
        self.annotation_path = annotation_path
        self.save_path = save_path

    def encode_label(self, labelmap):
        ret = np.ones_like(labelmap) * 255
        for idx, label in enumerate(self.mapping):

            ret[labelmap == label] = idx
        return ret.astype(np.uint8)

    def generate_label(self):
        train_path = os.path.join(self.annotation_path, 'train2017')
        val_path = os.path.join(self.annotation_path, 'val2017')
        save_train_path = os.path.join(self.save_path, 'train2017')
        save_val_path = os.path.join(self.save_path, 'val2017')

        if not os.path.exists(save_train_path):
            os.makedirs(save_train_path)
        if not os.path.exists(save_val_path):
            os.makedirs(save_val_path)

        for label_id in tqdm(os.listdir(train_path), desc='trainset'):
            label = np.array(
                Image.open(os.path.join(train_path, label_id)).convert('P'))
            label = self.encode_label(label)
            label = Image.fromarray(label)
            label.save(os.path.join(save_train_path, label_id))

        for label_id in tqdm(os.listdir(val_path), desc='valset'):
            label = np.array(
                Image.open(os.path.join(val_path, label_id)).convert('P'))
            label = self.encode_label(label)
            label = Image.fromarray(label)
            label.save(os.path.join(save_val_path, label_id))


def main():
    args = parse_args()
    generator = COCOStuffGenerator(
        annotation_path=args.annotation_path, save_path=args.save_path)
    generator.generate_label()


if __name__ == '__main__':
    main()
#/mnt/haoyuying/data/cocostuff/convert_annotations/val2017/000000086336.png

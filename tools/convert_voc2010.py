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
File: convert_voc2010.py
This file is based on https://www.cs.stanford.edu/~roozbeh/pascal-context/ to generate PASCAL-Context Dataset.
Before running, you should download the PASCAL VOC2010 from http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar, PASCAL-Context label id from https://www.cs.stanford.edu/~roozbeh/pascal-context/ and annotation file from https://codalabuser.blob.core.windows.net/public/trainval_merged.json. In segmentation map annotation for PascalContext, 0 stands for background, which is included in 60 categories. Then, make the folder structure as follow:

VOC2010
|
|--Annotations
|
|--ImageSets
|
|--SegmentationClass
|
|--JPEGImages
|
|--SegmentationObject
|
|--trainval_merged.json
"""

import argparse
import os

import tqdm
import numpy as np
from detail import Detail
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate PASCAL-Context dataset')
    parser.add_argument(
        '--voc_path', dest='voc_path', help='pascal voc path', type=str)
    parser.add_argument(
        '--annotation_path',
        dest='annotation_path',
        help='pascal context annotation path',
        type=str)

    return parser.parse_args()


class PascalContextGenerator(object):
    def __init__(self, voc_path, annotation_path):
        self.voc_path = voc_path
        self.annotation_path = annotation_path
        self.label_dir = os.path.join(self.voc_path, 'Context')
        self._image_dir = os.path.join(self.voc_path, 'JPEGImages')
        self.annFile = os.path.join(self.annotation_path,
                                    'trainval_merged.json')

        self._mapping = np.sort(
            np.array([
                0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22, 23, 397, 25,
                284, 158, 159, 416, 33, 162, 420, 454, 295, 296, 427, 44, 45,
                46, 308, 59, 440, 445, 31, 232, 65, 354, 424, 68, 326, 72, 458,
                34, 207, 80, 355, 85, 347, 220, 349, 360, 98, 187, 104, 105,
                366, 189, 368, 113, 115
            ]))
        self._key = np.array(range(len(self._mapping))).astype('uint8')

        self.train_detail = Detail(self.annFile, self._image_dir, 'train')
        self.train_ids = self.train_detail.getImgs()
        self.val_detail = Detail(self.annFile, self._image_dir, 'val')
        self.val_ids = self.val_detail.getImgs()

        if not os.path.exists(self.label_dir):
            os.makedirs(self.label_dir)

    def _class_to_index(self, mask, _mapping, _key):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert (values[i] in _mapping)
        index = np.digitize(mask.ravel(), _mapping, right=True)
        return _key[index].reshape(mask.shape)

    def save_mask(self, img_id, mode):
        if mode == 'train':
            mask = Image.fromarray(
                self._class_to_index(
                    self.train_detail.getMask(img_id),
                    _mapping=self._mapping,
                    _key=self._key))
        elif mode == 'val':
            mask = Image.fromarray(
                self._class_to_index(
                    self.val_detail.getMask(img_id),
                    _mapping=self._mapping,
                    _key=self._key))
        filename = img_id['file_name']
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".jpg"):
            mask_png_name = basename + '.png'
            mask.save(os.path.join(self.label_dir, mask_png_name))
        return basename

    def generate_label(self):

        with open(
                os.path.join(self.voc_path,
                             'ImageSets/Segmentation/train_context.txt'),
                'w') as f:
            for img_id in tqdm.tqdm(self.train_ids, desc='train'):
                basename = self.save_mask(img_id, 'train')
                f.writelines(''.join([basename, '\n']))

        with open(
                os.path.join(self.voc_path,
                             'ImageSets/Segmentation/val_context.txt'),
                'w') as f:
            for img_id in tqdm.tqdm(self.val_ids, desc='val'):
                basename = self.save_mask(img_id, 'val')
                f.writelines(''.join([basename, '\n']))

        with open(
                os.path.join(self.voc_path,
                             'ImageSets/Segmentation/trainval_context.txt'),
                'w') as f:
            for img in tqdm.tqdm(os.listdir(self.label_dir), desc='trainval'):
                if img.endswith('.png'):
                    basename = img.split('.', 1)[0]
                    f.writelines(''.join([basename, '\n']))


def main():
    args = parse_args()
    generator = PascalContextGenerator(
        voc_path=args.voc_path, annotation_path=args.annotation_path)
    generator.generate_label()


if __name__ == '__main__':
    main()

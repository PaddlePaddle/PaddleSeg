# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import os.path as osp
import shutil

import cv2

DEFECTS_LABELS = {
    'Free': 0,  # free is backgroud 
    'Blowhole': 1,
    'Break': 2,
    'Crack': 3,
    'Fray': 4,
    'Uneven': 5
}
TRAIN_PREFIX = ['exp0', 'exp1', 'exp2', 'exp3', 'exp4']
VAL_PREFIX = ['exp5', 'exp6']


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help="The directory of Magnetic-Tile dataset.")
    parser.add_argument(
        '--output_path',
        type=str,
        default="dataset/MT_dataset/",
        help="The directory for saving output dataset.")
    return parser.parse_args()


def _mkdir_p(path):
    """Make the path exists"""
    if not osp.exists(path):
        os.makedirs(path)


def _create_output_path(output_path):
    """
    Create output directories for training and validation images and annotations.
    """
    image_output_path = os.path.join(output_path, "images")
    anno_output_path = os.path.join(output_path, "anno")
    train_img_path = os.path.join(image_output_path, "train")
    val_img_path = os.path.join(image_output_path, "val")
    train_anno_path = os.path.join(anno_output_path, "train")
    val_anno_path = os.path.join(anno_output_path, "val")
    _mkdir_p(train_img_path)
    _mkdir_p(val_img_path)
    _mkdir_p(train_anno_path)
    _mkdir_p(val_anno_path)
    return [train_img_path, val_img_path, train_anno_path, val_anno_path]


def convert_MT(args):
    """
    Convert Magnetic-Tile dataset to paddleseg style data.
    """
    prefix = 'MT_'
    dataset_path, output_path = args.dataset_path, args.output_path
    train_img_path, val_img_path, train_anno_path, val_anno_path = _create_output_path(
        output_path)
    f_train = open(osp.join(output_path, 'train.txt'), "w")
    f_val = open(osp.join(output_path, 'val.txt'), "w")

    for name, label_id in DEFECTS_LABELS.items():
        folder_path = osp.join(dataset_path, prefix + name, 'Imgs')
        for img_name in os.listdir(folder_path):
            basename, ext = osp.splitext(img_name)
            if ext == '.png':
                continue

            anno_name = basename + '.png'
            anno_img = cv2.imread(osp.join(folder_path, anno_name), -1)
            anno_img[anno_img <= 240] = 0
            anno_img[anno_img > 240] = label_id
            exp_prefix = img_name.split('_')[0]

            if exp_prefix in TRAIN_PREFIX:
                shutil.copyfile(
                    osp.join(folder_path, img_name),
                    osp.join(train_img_path, img_name))
                cv2.imwrite(osp.join(train_anno_path, anno_name), anno_img)
                line = osp.join(train_img_path, img_name) + " " + osp.join(
                    train_anno_path, anno_name) + "\n"
                f_train.write(line)
            elif exp_prefix in VAL_PREFIX:
                shutil.copyfile(
                    osp.join(folder_path, img_name),
                    osp.join(val_img_path, img_name))
                cv2.imwrite(osp.join(val_anno_path, anno_name), anno_img)
                line = osp.join(val_img_path, img_name) + " " + osp.join(
                    val_anno_path, anno_name) + "\n"
                f_val.write(line)
            else:
                raise ValueError(
                    'img name can not match any train or val prefix.')
    f_train.close()
    f_val.close()


if __name__ == "__main__":
    args = parse_args()
    convert_MT(args)

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


import argparse
import os
import random
import shutil
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def get_image(fp):
    image = np.array(Image.open(fp))
    image = image.astype('float32')
    image = image / np.max(image) * 255
    image = np.tile(image[..., None], [1, 1, 3])
    image = image.astype('uint8')
    return image


def to_image_id(image_filepath):
    image_dirs = image_filepath.replace('/', '\\').split('\\')
    image_dirs = [image_dirs[2]] + image_dirs[4].split('_')[:2]
    image_id = '_'.join(image_dirs)
    return image_id


def rle_decode(mask_rle, image_shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(image_shape[0] * image_shape[1], dtype='uint8')
    for low, high in zip(starts, ends):
        img[low:high] = 1
    return img.reshape(image_shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help="the directory of original UWMGI dataset zip file",
        type=str)
    parser.add_argument(
        "output",
        help="the directory to save converted UWMGI dataset",
        type=str)
    parser.add_argument(
        '--train_proportion',
        help='the proportion of train dataset',
        type=float,
        default=0.8)
    parser.add_argument(
        '--val_proportion',
        help='the proportion of validation dataset',
        type=float,
        default=0.2)
    args = parser.parse_args()

    assert os.path.exists(args.input), \
        f"The directory({args.input}) of " \
        f"original UWMGI dataset does not exist!"
    assert zipfile.is_zipfile(args.input)

    assert 0 < args.train_proportion <= 1
    assert 0 <= args.val_proportion < 1
    assert args.train_proportion + args.val_proportion == 1

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    else:
        if os.listdir(args.output):
            shutil.rmtree(args.output)
    os.makedirs(os.path.join(args.output, 'images/train'))
    os.makedirs(os.path.join(args.output, 'annotations/train'))
    os.makedirs(os.path.join(args.output, 'images/val'))
    os.makedirs(os.path.join(args.output, 'annotations/val'))

    with zipfile.ZipFile(args.input, 'r') as zip_fp:
        total_df = pd.read_csv(zip_fp.open('train.csv', 'r'))

        total_image_namelist = []
        for name in zip_fp.namelist():
            if os.path.splitext(name)[1] == '.png':
                total_image_namelist.append(name)
        train_image_namelist = random.sample(
            total_image_namelist, int(
                len(total_image_namelist) * args.train_proportion))
        val_image_namelist = np.setdiff1d(
            total_image_namelist, train_image_namelist)

        pbar = tqdm(total=len(total_image_namelist))
        for image_namelist, split in zip(
                [train_image_namelist, val_image_namelist], ['train', 'val']):
            txt_lines = []
            for image_name in image_namelist:
                with zip_fp.open(image_name, 'r') as fp:
                    image = get_image(fp)
                    image_id = to_image_id(image_name)
                    anns = total_df[total_df['id'] == image_id]
                    height, width = image.shape[:2]
                    mask = np.zeros([height, width * 3], dtype='uint8')
                    for _, ann in anns.iterrows():
                        if not pd.isna(ann['segmentation']):
                            if ann['class'] == 'large_bowel':
                                mask[:, 0:width] = rle_decode(
                                    ann['segmentation'], (height, width))
                            elif ann['class'] == 'small_bowel':
                                mask[:, width:width * 2] = rle_decode(
                                    ann['segmentation'], (height, width))
                            else:  # ann['class'] == 'stomach'
                                mask[:, width * 2:] = rle_decode(
                                    ann['segmentation'], (height, width))
                    cv2.imwrite(os.path.join(
                        args.output, 'images', split, image_id + '.jpg'), image)
                    cv2.imwrite(os.path.join(
                        args.output, 'annotations', split, image_id + '.png'), mask)
                    txt_lines.append(
                        os.path.join('images', split, image_id + '.jpg')
                        + ' ' + os.path.join('annotations', split, image_id + '.png'))
                    pbar.update()

            with open(os.path.join(args.output, split + '.txt'), 'w') as fp:
                fp.write('\n'.join(txt_lines))


if __name__ == '__main__':
    main()

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
"""
File: convert_multilabel.py
This file is used to convert `uwmgi` or `coco` type dataset to support multi-label dataset format.
Examples of usage are as follows:
1. convert UWMGI dataset
python convert_multilabel.py --dataset_type uwmgi --zip_input ${uwmgi_origin_zip_file} --output ${save_dir} --train_proportion 0.8 --val_proportion 0.2
2. convert COCO type dataset
2.1 not yet split training and validation dataset
python convert_multilabel.py --dataset_type coco --img_input ${img_dir} --ann_input ${ann_dir} --output ${save_dir} --train_proportion 0.8 --val_proportion 0.2
2.2 training and validation dataset split
python convert_multilabel.py --dataset_type coco --img_input ${train_img_dir} --ann_input ${train_ann_dir} --output ${save_dir} --train_proportion 1.0 --val_proportion 0.0
python convert_multilabel.py --dataset_type coco --img_input ${val_img_dir} --ann_input ${val_ann_dir} --output ${save_dir} --train_proportion 0.0 --val_proportion 1.0
"""

import argparse
import os
import random
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


def uwmgi_get_image(fp):
    image = np.array(Image.open(fp))
    image = image.astype('float32')
    image = image / np.max(image) * 255
    image = np.tile(image[..., None], [1, 1, 3])
    image = image.astype('uint8')
    return image


def uwmgi_get_image_id(image_filepath):
    try:
        image_dirs = image_filepath.split('/')
        image_dirs = [image_dirs[1]] + image_dirs[3].split('_')[:2]
    except:
        image_dirs = image_filepath.replace('/', '\\').split('\\')
        image_dirs = [image_dirs[2]] + image_dirs[4].split('_')[:2]
    image_id = '_'.join(image_dirs)
    return image_id


def uwmgi_rle_decode(mask_rle, image_shape):
    s = mask_rle.split()
    starts, lengths = [
        np.asarray(
            x, dtype=int) for x in (s[0:][::2], s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(image_shape[0] * image_shape[1], dtype='uint8')
    for low, high in zip(starts, ends):
        img[low:high] = 1
    return img.reshape(image_shape)


def uwmgi_to_multilabel_format(args):
    with zipfile.ZipFile(args.zip_input, 'r') as zip_fp:
        total_df = pd.read_csv(zip_fp.open('train.csv', 'r'))

        total_image_namelist = []
        for name in zip_fp.namelist():
            if os.path.splitext(name)[1] == '.png':
                total_image_namelist.append(name)
        train_image_namelist = random.sample(
            total_image_namelist,
            int(len(total_image_namelist) * args.train_proportion))
        val_image_namelist = np.setdiff1d(total_image_namelist,
                                          train_image_namelist)

        pbar = tqdm(total=len(total_image_namelist))
        for image_namelist, split in zip(
            [train_image_namelist, val_image_namelist], ['train', 'val']):
            txt_lines = []
            for image_name in image_namelist:
                with zip_fp.open(image_name, 'r') as fp:
                    image = uwmgi_get_image(fp)
                    image_id = uwmgi_get_image_id(image_name)
                    anns = total_df[total_df['id'] == image_id]
                    height, width = image.shape[:2]
                    mask = np.zeros([height, width * 3], dtype='uint8')
                    for _, ann in anns.iterrows():
                        if not pd.isna(ann['segmentation']):
                            if ann['class'] == 'large_bowel':
                                mask[:, 0:width] = uwmgi_rle_decode(
                                    ann['segmentation'], (height, width))
                            elif ann['class'] == 'small_bowel':
                                mask[:, width:width * 2] = uwmgi_rle_decode(
                                    ann['segmentation'], (height, width))
                            else:  # ann['class'] == 'stomach'
                                mask[:, width * 2:] = uwmgi_rle_decode(
                                    ann['segmentation'], (height, width))
                    cv2.imwrite(
                        os.path.join(args.output, 'images', split,
                                     image_id + '.jpg'), image)
                    cv2.imwrite(
                        os.path.join(args.output, 'annotations', split,
                                     image_id + '.png'), mask)
                    txt_lines.append(
                        os.path.join('images', split, image_id + '.jpg') + ' ' +
                        os.path.join('annotations', split, image_id + '.png'))
                    pbar.update()

            with open(os.path.join(args.output, split + '.txt'), 'w') as fp:
                fp.write('\n'.join(txt_lines))


def coco_to_multilabel_format(args):
    coco = COCO(args.ann_input)
    cat_id_map = {
        old_cat_id: new_cat_id
        for new_cat_id, old_cat_id in enumerate(coco.getCatIds())
    }
    num_classes = len(list(cat_id_map.keys()))

    assert 'annotations' in coco.dataset, \
        'Annotation file: {} does not contains ground truth!!!'.format(args.ann_input)

    total_img_id_list = sorted(list(coco.imgToAnns.keys()))
    train_img_id_list = random.sample(
        total_img_id_list, int(len(total_img_id_list) * args.train_proportion))
    val_img_id_list = np.setdiff1d(total_img_id_list, train_img_id_list)

    pbar = tqdm(total=len(total_img_id_list))
    for img_id_list, split in zip([train_img_id_list, val_img_id_list],
                                  ['train', 'val']):
        txt_lines = []
        for img_id in img_id_list:
            img_info = coco.loadImgs([img_id])[0]
            img_filename = img_info['file_name']
            img_w = img_info['width']
            img_h = img_info['height']

            img_filepath = os.path.join(args.img_input, img_filename)
            if not os.path.exists(img_filepath):
                print('Illegal image file: {}, '
                      'and it will be ignored'.format(img_filepath))
                continue

            if img_w < 0 or img_h < 0:
                print('Illegal width: {} or height: {} in annotation, '
                      'and im_id: {} will be ignored'.format(img_w, img_h,
                                                             img_id))
                continue

            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)

            mask = np.zeros([img_h, num_classes * img_w], dtype='uint8')
            for ann in anns:
                cat_id = cat_id_map[ann['category_id']]
                one_cls_mask = coco.annToMask(ann)
                mask[:, cat_id * img_w:(cat_id + 1) * img_w] = np.where(
                    one_cls_mask, one_cls_mask,
                    mask[:, cat_id * img_w:(cat_id + 1) * img_w])

            image = cv2.imread(img_filepath, cv2.IMREAD_COLOR)
            cv2.imwrite(
                os.path.join(args.output, 'images', split,
                             os.path.splitext(img_filename)[0] + '.jpg'), image)
            cv2.imwrite(
                os.path.join(args.output, 'annotations', split,
                             os.path.splitext(img_filename)[0] + '.png'), mask)
            txt_lines.append(
                os.path.join('images', split,
                             os.path.splitext(img_filename)[0] + '.jpg') + ' ' +
                os.path.join('annotations', split,
                             os.path.splitext(img_filename)[0] + '.png'))
            pbar.update()

        with open(os.path.join(args.output, split + '.txt'), 'w') as fp:
            fp.write('\n'.join(txt_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_type',
        help='the type of dataset, can be `uwmgi` or `coco`',
        type=str)
    parser.add_argument(
        "--zip_input",
        help="the directory of original dataset zip file",
        type=str)
    parser.add_argument(
        "--img_input",
        help="the directory of original dataset image file",
        type=str)
    parser.add_argument(
        "--ann_input",
        help="the directory of original dataset annotation file",
        type=str)
    parser.add_argument(
        "--output", help="the directory to save converted dataset", type=str)
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

    assert args.dataset_type in ['uwmgi', 'coco'], \
        "Now only support the `uwmgi` and `coco`!!!"

    assert 0 <= args.train_proportion <= 1
    assert 0 <= args.val_proportion <= 1
    assert args.train_proportion + args.val_proportion == 1

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    os.makedirs(os.path.join(args.output, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'annotations/train'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'annotations/val'), exist_ok=True)

    if args.dataset_type == 'uwmgi':
        assert os.path.exists(args.zip_input), \
            f"The directory({args.zip_input}) of " \
            f"original UWMGI dataset does not exist!"
        assert zipfile.is_zipfile(args.zip_input)

        uwmgi_to_multilabel_format(args)

    else:  # args.dataset_type == 'coco'
        assert os.path.exists(args.img_input), \
            f"The directory({args.img_input}) of " \
            f"original image file does not exist!"
        assert os.path.exists(args.ann_input), \
            f"The directory({args.ann_input}) of " \
            f"original annotation file does not exist!"

        coco_to_multilabel_format(args)

    print("Dataset converts success, the data path: {}".format(args.output))


if __name__ == '__main__':
    main()

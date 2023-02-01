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
# Adapted from https://github.com/open-mmlab/mmsegmentation/blob/master/tools/convert_datasets/isaid.py
#
# Original copyright info:
# Copyright (c) OpenMMLab. All rights reserved.
#
# See original LICENSE at https://github.com/open-mmlab/mmsegmentation/blob/master/LICENSE

import argparse
import glob
import os
import os.path as osp
import shutil
import tempfile
import zipfile

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

iSAID_palette = \
    {
        0: (0, 0, 0),
        1: (0, 0, 63),
        2: (0, 63, 63),
        3: (0, 63, 0),
        4: (0, 63, 127),
        5: (0, 63, 191),
        6: (0, 63, 255),
        7: (0, 127, 63),
        8: (0, 127, 127),
        9: (0, 0, 127),
        10: (0, 0, 191),
        11: (0, 0, 255),
        12: (0, 191, 127),
        13: (0, 127, 191),
        14: (0, 127, 255),
        15: (0, 100, 155)
    }

iSAID_invert_palette = {v: k for k, v in iSAID_palette.items()}


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def iSAID_convert_from_color(arr_3d, palette=iSAID_invert_palette):
    """RGB-color encoding to grayscale labels."""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def pad(img, shape=None, padding=None, pad_val=0, padding_mode='constant'):
    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        width = max(shape[1] - img.shape[1], 0)
        height = max(shape[0] - img.shape[0], 0)
        padding = (0, 0, width, height)

    # Check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # Check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # Check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)

    return img


def slide_crop_image(src_path, out_dir, mode, patch_H, patch_W, overlap):
    img = np.asarray(Image.open(src_path).convert('RGB'))

    img_H, img_W, _ = img.shape

    if img_H < patch_H and img_W > patch_W:

        img = pad(img, shape=(patch_H, img_W), pad_val=0)

        img_H, img_W, _ = img.shape

    elif img_H > patch_H and img_W < patch_W:

        img = pad(img, shape=(img_H, patch_W), pad_val=0)

        img_H, img_W, _ = img.shape

    elif img_H < patch_H and img_W < patch_W:

        img = pad(img, shape=(patch_H, patch_W), pad_val=0)

        img_H, img_W, _ = img.shape

    for x in range(0, img_W, patch_W - overlap):
        for y in range(0, img_H, patch_H - overlap):
            x_str = x
            x_end = x + patch_W
            if x_end > img_W:
                diff_x = x_end - img_W
                x_str -= diff_x
                x_end = img_W
            y_str = y
            y_end = y + patch_H
            if y_end > img_H:
                diff_y = y_end - img_H
                y_str -= diff_y
                y_end = img_H

            img_patch = img[y_str:y_end, x_str:x_end, :]
            img_patch = Image.fromarray(img_patch.astype(np.uint8))
            image = osp.basename(src_path).split('.')[0] + '_' + str(
                y_str) + '_' + str(y_end) + '_' + str(x_str) + '_' + str(
                    x_end) + '.png'
            # print(image)
            save_path_image = osp.join(out_dir, 'img_dir', mode, str(image))
            img_patch.save(save_path_image)


def slide_crop_label(src_path, out_dir, mode, patch_H, patch_W, overlap):
    label = Image.open(src_path).convert('RGB')
    label = np.asarray(label)
    label = iSAID_convert_from_color(label)
    img_H, img_W = label.shape

    if img_H < patch_H and img_W > patch_W:

        label = pad(label, shape=(patch_H, img_W), pad_val=255)

        img_H = patch_H

    elif img_H > patch_H and img_W < patch_W:

        label = pad(label, shape=(img_H, patch_W), pad_val=255)

        img_W = patch_W

    elif img_H < patch_H and img_W < patch_W:

        label = pad(label, shape=(patch_H, patch_W), pad_val=255)

        img_H = patch_H
        img_W = patch_W

    for x in range(0, img_W, patch_W - overlap):
        for y in range(0, img_H, patch_H - overlap):
            x_str = x
            x_end = x + patch_W
            if x_end > img_W:
                diff_x = x_end - img_W
                x_str -= diff_x
                x_end = img_W
            y_str = y
            y_end = y + patch_H
            if y_end > img_H:
                diff_y = y_end - img_H
                y_str -= diff_y
                y_end = img_H

            lab_patch = label[y_str:y_end, x_str:x_end]
            lab_patch = Image.fromarray(lab_patch.astype(np.uint8))

            image = osp.basename(src_path).split('.')[0].split('_')[
                0] + '_' + str(y_str) + '_' + str(y_end) + '_' + str(
                    x_str) + '_' + str(x_end) + '_instance_color_RGB' + '.png'
            lab_patch.save(osp.join(out_dir, 'ann_dir', mode, str(image)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='Path of raw iSAID dataset.')
    parser.add_argument('--tmp_dir', help='Path of the temporary directory.')
    parser.add_argument('-o', '--out_dir', help='Output path.')

    parser.add_argument(
        '--patch_width',
        default=896,
        type=int,
        help='Width of the cropped image patch.')
    parser.add_argument(
        '--patch_height',
        default=896,
        type=int,
        help='Height of the cropped image patch.')
    parser.add_argument(
        '--overlap_area', default=384, type=int, help='Overlap area.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset_path = args.dataset_path
    # image patch width and height
    patch_H, patch_W = args.patch_width, args.patch_height

    overlap = args.overlap_area  # overlap area

    if args.out_dir is None:
        out_dir = osp.join('data', 'iSAID')
    else:
        out_dir = args.out_dir

    print('Creating directories...')
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'test'))

    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'test'))

    assert os.path.exists(os.path.join(dataset_path, 'train')), \
        'train is not in {}'.format(dataset_path)
    assert os.path.exists(os.path.join(dataset_path, 'val')), \
        'val is not in {}'.format(dataset_path)
    assert os.path.exists(os.path.join(dataset_path, 'test')), \
        'test is not in {}'.format(dataset_path)

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for dataset_mode in ['train', 'val', 'test']:

            # for dataset_mode in [ 'test']:
            print('Extracting  {}ing.zip...'.format(dataset_mode))
            img_zipp_list = glob.glob(
                os.path.join(dataset_path, dataset_mode, 'images', '*.zip'))
            print('Find the data', img_zipp_list)
            for img_zipp in img_zipp_list:
                zip_file = zipfile.ZipFile(img_zipp)
                zip_file.extractall(os.path.join(tmp_dir, dataset_mode, 'img'))
            src_path_list = glob.glob(
                os.path.join(tmp_dir, dataset_mode, 'img', 'images', '*.png'))

            for i, img_path in enumerate(tqdm(src_path_list)):
                if dataset_mode != 'test':
                    slide_crop_image(img_path, out_dir, dataset_mode, patch_H,
                                     patch_W, overlap)

                else:
                    shutil.move(img_path,
                                os.path.join(out_dir, 'img_dir', dataset_mode))

            if dataset_mode != 'test':
                label_zipp_list = glob.glob(
                    os.path.join(dataset_path, dataset_mode, 'Semantic_masks',
                                 '*.zip'))
                for label_zipp in label_zipp_list:
                    zip_file = zipfile.ZipFile(label_zipp)
                    zip_file.extractall(
                        os.path.join(tmp_dir, dataset_mode, 'lab'))

                lab_path_list = glob.glob(
                    os.path.join(tmp_dir, dataset_mode, 'lab', 'images',
                                 '*.png'))
                for i, lab_path in enumerate(tqdm(lab_path_list)):
                    slide_crop_label(lab_path, out_dir, dataset_mode, patch_H,
                                     patch_W, overlap)

        print('Removing the temporary files...')

    print('Done!')

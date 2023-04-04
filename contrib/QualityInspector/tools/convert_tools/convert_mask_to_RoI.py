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
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)

import cv2
import numpy as np

from qinspector.utils.bbox_utils import adjust_bbox


def get_args():
    parser = argparse.ArgumentParser(
        description='Mask convert to RoI for Segmentation')
    # Parameters
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='The directory of images.')
    parser.add_argument(
        '--anno_path',
        type=str,
        required=True,
        help='The directory of ground truth masks.')
    parser.add_argument(
        '--class_num',
        type=int,
        required=True,
        help='Number of categories, without background.')
    parser.add_argument(
        '--to_binary',
        action='store_true',
        help='Whether or not to convert mask to binary RoI.')
    parser.add_argument(
        '--suffix',
        type=str,
        default='.png',
        help='The suffix of filename between gt and image.')
    parser.add_argument(
        '--pad_scale',
        type=float,
        default=0.5,
        help='The padding scale of box to crop image.')
    parser.add_argument(
        '--output_path',
        type=str,
        default='./output/RoI/',
        help='The directory for saving the RoI images, masks and list.txt.')
    return parser.parse_args()


def _mkdir_p(path):
    """Make the path exists"""
    if not osp.exists(path):
        os.makedirs(path)


def convert_mask_to_roi(args):
    image_path = args.image_path
    anno_path = args.anno_path
    suffix = args.suffix
    output_path = args.output_path

    output_image_path = osp.join(output_path, 'images')
    _mkdir_p(output_image_path)
    output_anno_path = osp.join(output_path, 'anno')
    _mkdir_p(output_anno_path)

    classid_list = list(range(1, args.class_num + 1))
    file_list = osp.join(output_path, 'RoI.txt')
    f = open(file_list, "w")

    for img_name in os.listdir(image_path):
        file_name = osp.join(image_path, img_name)
        img = cv2.imread(file_name)

        if img is None:
            raise FileNotFoundError('{} is not found.'.format(
                osp.join(file_name)))

        base_name = img_name.split('.')[0]
        anno_name = osp.join(anno_path, base_name + suffix)
        mask = cv2.imread(anno_name, -1)

        if mask is None:
            raise FileNotFoundError('{} is not found.'.format(
                osp.join(anno_name)))
        if mask.sum() == 0:
            continue

        for cls_id in classid_list:
            class_map = np.equal(mask, cls_id).astype(np.uint8)
            if class_map.sum() == 0:
                continue
            _, _, stats, _ = cv2.connectedComponentsWithStats(class_map)
            for i, stat in enumerate(stats):
                if i == 0:
                    continue
                stat = list(map(int, stat))
                bbox = adjust_bbox(
                    [stat[0], stat[1], stat[0] + stat[2], stat[1] + stat[3]],
                    img.shape[:2],
                    pad_scale=args.pad_scale)
                img_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                mask_crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if args.to_binary:
                    mask_crop[mask_crop > 0] = 1
                # save crop img, mask and write to txt file
                img_save_path = osp.join(output_image_path,
                                         f'{base_name}_{cls_id}_{i}.png')
                anno_save_path = osp.join(output_anno_path,
                                          f'{base_name}_{cls_id}_{i}{suffix}')
                cv2.imwrite(img_save_path, img_crop)
                cv2.imwrite(anno_save_path, mask_crop)
                line = img_save_path + " " + anno_save_path + '\n'
                f.write(line)
    f.close()


if __name__ == '__main__':
    args = get_args()
    convert_mask_to_roi(args)

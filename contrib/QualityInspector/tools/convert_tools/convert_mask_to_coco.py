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
import json
import os
import os.path as osp

import cv2
import numpy as np


def get_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(
        description='Mask convert to Json for detection')
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
        '--label_file',
        type=str,
        default=None,
        help='The path of a json file which gives class name and category id.')
    parser.add_argument(
        '--suffix',
        type=str,
        default='.png',
        help='The suffix of filename between gt and image.')
    parser.add_argument(
        '--output_name',
        type=str,
        default='coco.json',
        help='The file name for saving the output json file.')
    return parser.parse_args()


def convert_mask_to_coco(args):
    """convert mask to coco format """
    images, annotations = [], []
    image_id, anno_id = 0, 0
    classid_list = list(range(1, args.class_num + 1))  # 0 is background
    for img_name in os.listdir(args.image_path):
        image_id += 1
        file_name = osp.join(args.image_path, img_name)
        img = cv2.imread(file_name)
        if img is None:
            raise FileNotFoundError('{} is not found.'.format(
                osp.join(file_name)))
        image_info = {
            "file_name": file_name,
            "id": image_id,
            "width": img.shape[1],
            "height": img.shape[0],
        }
        images.append(image_info)
        basename = osp.splitext(img_name)[0]
        anno_name = osp.join(args.anno_path, basename + args.suffix)
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
            _, labels, stats, _ = cv2.connectedComponentsWithStats(class_map)
            for i, stat in enumerate(stats):
                if i == 0:
                    continue  # skip background
                anno = {}
                polygon = []
                stat = list(map(int, stat))
                contours, _ = cv2.findContours((labels == i).astype(np.uint8),
                                               cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)
                polygon.append(contours[0].flatten().astype(int).tolist())
                anno_id += 1
                anno = {
                    "id": anno_id,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "segmentation": polygon,
                    "area": stat[2] * stat[3],
                    "bbox": [stat[0], stat[1], stat[2], stat[3]],
                    "iscrowd": 0,
                }
                annotations.append(anno)

    categories = [{
        "supercategory": "defect",
        "id": cls_id,
        "name": str(cls_id)
    } for cls_id in classid_list]
    if args.label_file:
        id_to_label = json.load(open(args.label_file))
        for category in categories:
            category["name"] = id_to_label[str(category["id"])]

    json_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(args.output_name, "w") as f:
        json.dump(json_data, f, indent=2)


if __name__ == '__main__':
    args = get_args()
    convert_mask_to_coco(args)

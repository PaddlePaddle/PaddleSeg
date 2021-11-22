#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os

import cv2
import json
import argparse
import numpy as np

TRAIN_DATA_SET = [
    'label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'
]
TEST_DATA_SET = ['test_label.json']

Image_Height, Image_Width = 720, 1280
# lane width (pixel)
LANE_SEG_WIDTH = 30
# has 7 class instances
CLASS_NUMS = 7


def generate_seg_label_image(label):
    # this code copy from https://github.com/ZJULearning/resa/blob/main/tools/generate_seg_tusimple.py
    lanes = []
    _lanes = []
    slope = []  # identify 0th, 1st, 2nd, 3rd, 4th, 5th lane through slope
    for i in range(len(label['lanes'])):
        l = [(x, y) for x, y in zip(label['lanes'][i], label['h_samples'])
             if x >= 0]
        if (len(l) > 1):
            _lanes.append(l)
            slope.append(
                np.arctan2(l[-1][1] - l[0][1], l[0][0] - l[-1][0]) / np.pi *
                180)
    _lanes = [_lanes[i] for i in np.argsort(slope)]
    slope = [slope[i] for i in np.argsort(slope)]

    idx = [None for i in range(CLASS_NUMS - 1)]
    for i in range(len(slope)):
        if slope[i] <= 90:
            idx[2] = i
            idx[1] = i - 1 if i > 0 else None
            idx[0] = i - 2 if i > 1 else None
        else:
            idx[3] = i
            idx[4] = i + 1 if i + 1 < len(slope) else None
            idx[5] = i + 2 if i + 2 < len(slope) else None
            break
    for i in range(CLASS_NUMS - 1):
        lanes.append([] if idx[i] is None else _lanes[idx[i]])

    seg_img = np.zeros([Image_Height, Image_Width], np.uint8)

    for i in range(len(lanes)):
        coords = lanes[i]
        if len(coords) < 4:
            continue
        for j in range(len(coords) - 1):
            cv2.line(seg_img, coords[j], coords[j + 1], i + 1,
                     LANE_SEG_WIDTH // 2)

    return seg_img


def generate_labels(args, src_dir, label_dir, image_set, mode):
    os.makedirs(os.path.join(args.root, src_dir, label_dir), exist_ok=True)
    label_file = open(
        os.path.join(args.root, src_dir, "{}_list.txt".format(mode)), "w")
    for json_name in (image_set):
        with open(os.path.join(args.root, src_dir, json_name)) as jsonfile:
            for jsonline in jsonfile:
                label = json.loads(jsonline)
                seg_img = generate_seg_label_image(label)
                img_path = label['raw_file']
                seg_path = img_path.split("/")
                seg_path, img_name = os.path.join(args.root, src_dir, label_dir,
                                                  seg_path[1],
                                                  seg_path[2]), seg_path[3]
                os.makedirs(seg_path, exist_ok=True)
                seg_path = os.path.join(seg_path, img_name[:-3] + "png")
                cv2.imwrite(seg_path, seg_img)

                img_path = "/".join([src_dir, img_path])
                seg_path = "/".join([
                    src_dir, label_dir, *img_path.split("/")[2:4],
                    img_name[:-3] + "png"
                ])
                if seg_path[0] != '/':
                    seg_path = '/' + seg_path
                if img_path[0] != '/':
                    img_path = '/' + img_path

                label_str = []
                label_str.insert(0, seg_path)
                label_str.insert(0, img_path)
                label_str = " ".join(label_str) + "\n"
                label_file.write(label_str)


def process_tusimple_dataset(args):
    print("generating train set...")
    generate_labels(args, "train_set", "labels", TRAIN_DATA_SET, mode="train")
    print("generating test set...")
    generate_labels(args, "test_set", "labels", TEST_DATA_SET, mode="test")
    print("generate finish!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=str,
        default=None,
        help='The origin path of unzipped tusimple dataset')
    args = parser.parse_args()

    process_tusimple_dataset(args)

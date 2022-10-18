# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import print_function

import argparse
import glob
import math
import json
import os
import os.path as osp
import numpy as np
import PIL.Image
import PIL.ImageDraw
import cv2

from gray2pseudo_color import get_color_map_list


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help='input annotated directory')
    return parser.parse_args()


def main(args):
    output_dir = osp.join(args.input_dir, 'annotations')
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        print('Creating annotations directory:', output_dir)

    # get the all class names for the given dataset
    class_names = ['_background_']
    for label_file in glob.glob(osp.join(args.input_dir, '*.json')):
        with open(label_file) as f:
            data = json.load(f)
            for shape in data['shapes']:
                label = shape['label']
                cls_name = label
                if not cls_name in class_names:
                    class_names.append(cls_name)

    class_name_to_id = {}
    for i, class_name in enumerate(class_names):
        class_id = i  # starts with 0
        class_name_to_id[class_name] = class_id
        if class_id == 0:
            assert class_name == '_background_'
    class_names = tuple(class_names)
    print('class_names:', class_names)

    out_class_names_file = osp.join(args.input_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    color_map = get_color_map_list(256)

    for label_file in glob.glob(osp.join(args.input_dir, '*.json')):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            out_png_file = osp.join(output_dir, base + '.png')

            data = json.load(f)

            img_file = osp.join(osp.dirname(label_file), data['imagePath'])
            img = np.asarray(cv2.imread(img_file))

            lbl = shape2label(
                img_size=img.shape,
                shapes=data['shapes'],
                class_name_mapping=class_name_to_id, )

            if osp.splitext(out_png_file)[1] != '.png':
                out_png_file += '.png'
            # Assume label ranges [0, 255] for uint8,
            if lbl.min() >= 0 and lbl.max() <= 255:
                lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
                lbl_pil.putpalette(color_map)
                lbl_pil.save(out_png_file)
            else:
                raise ValueError(
                    '[%s] Cannot save the pixel-wise class label as PNG. '
                    'Please consider using the .npy format.' % out_png_file)


def shape2mask(img_size, points):
    label_mask = PIL.Image.fromarray(np.zeros(img_size[:2], dtype=np.uint8))
    image_draw = PIL.ImageDraw.Draw(label_mask)
    points_list = [tuple(point) for point in points]
    assert len(points_list) > 2, 'Polygon must have points more than 2'
    image_draw.polygon(xy=points_list, outline=1, fill=1)
    return np.array(label_mask, dtype=bool)


def shape2label(img_size, shapes, class_name_mapping):
    label = np.zeros(img_size[:2], dtype=np.int32)
    for shape in shapes:
        points = shape['points']
        class_name = shape['label']
        shape_type = shape.get('shape_type', None)
        class_id = class_name_mapping[class_name]
        label_mask = shape2mask(img_size[:2], points)
        label[label_mask] = class_id
    return label


if __name__ == '__main__':
    args = parse_args()
    main(args)

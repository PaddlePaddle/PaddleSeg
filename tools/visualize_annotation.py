# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
visualize the annoted images:
1. Add the origin image and annotated image to produce the weighted image.
2. Paste the origin image and weighted image to generate the final image.
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from paddleseg import utils
from paddleseg.utils import logger, progbar, visualize


def parse_args():
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument(
        '--file_path',
        help='The file contains the path of origin and annotated images',
        type=str)
    parser.add_argument('--save_dir',
                        help='The directory for saving the visualized images',
                        type=str,
                        default='./output/visualize_annotation')
    return parser.parse_args()


def get_images_path(file_path):
    """
    Get the path of origin images and annotated images.
    """
    assert os.path.isfile(file_path)

    images_path = []
    image_dir = os.path.dirname(file_path)

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            origin_path, annot_path = line.split(" ")
            origin_path = os.path.join(image_dir, origin_path)
            annot_path = os.path.join(image_dir, annot_path)
            images_path.append([origin_path, annot_path])

    return images_path


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    images_path = get_images_path(args.file_path)
    bar = progbar.Progbar(target=len(images_path), verbose=1)

    for idx, (origin_path, annot_path) in enumerate(images_path):
        origin_img = Image.open(origin_path)
        annot_img = Image.open(annot_path)
        annot_img = np.array(annot_img)

        bar.update(idx + 1)
        if len(np.unique(annot_img)) == 1:
            continue

        # weighted image
        color_map = visualize.get_color_map_list(256)
        weighted_img = utils.visualize.visualize(origin_path,
                                                 annot_img,
                                                 color_map,
                                                 weight=0.6)
        weighted_img = Image.fromarray(
            cv2.cvtColor(weighted_img, cv2.COLOR_BGR2RGB))

        # result image
        result_img = visualize.paste_images([origin_img, weighted_img])

        # save
        image_name = os.path.split(origin_path)[-1]
        result_img.save(os.path.join(args.save_dir, image_name))


if __name__ == '__main__':
    args = parse_args()
    main(args)
